import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

def load_data(data_dir='data'):
    train = pd.read_csv(f'{data_dir}/train_gpt.csv')
    val = pd.read_csv(f'{data_dir}/val_gpt.csv')
    test = pd.read_csv(f'{data_dir}/test_gpt.csv')
    return train, val, test

def precision_at_k(y_true, y_prob, k_percent):
    k = int(len(y_true) * (k_percent / 100))
    if k == 0: return 0.0
    
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    top_k = df.head(k)
    return top_k['true'].mean()

def train_xgboost_optuna():
    # 1. Load
    train, val, test = load_data()
    
    # 2. Prepare cols
    target = 'lapse_next_3m'
    drop_cols = ['policy_id', 'month', 'split', 'post_event_notice_sent', target]
    
    features = [c for c in train.columns if c not in drop_cols]
    
    print(f"Features: {features}")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    
    # Feature binning into intervals
    def bin_features(df):
        df = df.copy()
        
        # Age binning into 4 intervals
        df['age'] = pd.cut(df['age'], 
                          bins=[0, 30, 45, 60, 150], 
                          labels=['18-30', '31-45', '46-60', '61+'],
                          right=True)
        
        # Premium binning into 5 intervals
        df['premium'] = pd.cut(df['premium'], 
                              bins=[0, 100, 150, 200, 300, 10000], 
                              labels=['<100', '100-150', '150-200', '200-300', '300+'],
                              right=True)
        
        # Tenure binning into 5 intervals (months)
        df['tenure_m'] = pd.cut(df['tenure_m'], 
                               bins=[-1, 6, 12, 24, 48, 10000], 
                               labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+'],
                               right=True)
        return df
    
    train = bin_features(train)
    val = bin_features(val)
    test = bin_features(test)
    
    # Cat encoding - use single encoder for all categorical features
    cat_cols = ['region', 'age', 'premium', 'tenure_m'] # All binned features
    
    # Fit one encoder for all categorical columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(train[cat_cols])
    
    # Transform all datasets
    train[cat_cols] = encoder.transform(train[cat_cols])
    val[cat_cols] = encoder.transform(val[cat_cols])
    test[cat_cols] = encoder.transform(test[cat_cols])


    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]
    
    # 3. Optuna
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'logloss' # using logloss as proxy for general perf, custom pr-auc is slower
        }
        
        # XGBClassifier with early stopping
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = clf.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds)
        return score

    print("Starting Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("Best params:", study.best_params)
    
    # 4. Final Train
    best_params = study.best_params
    best_params['n_estimators'] = 1000
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['early_stopping_rounds'] = 50
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 5. Evaluate on Test
    probs = model.predict_proba(X_test)[:, 1]
    
    auc_pr = average_precision_score(y_test, probs)
    p_at_1 = precision_at_k(y_test, probs, 1)
    p_at_5 = precision_at_k(y_test, probs, 5)
    
    metrics = {
        'auc_pr': auc_pr,
        'precision_at_1_pct': p_at_1,
        'precision_at_5_pct': p_at_5
    }
    print("Test Metrics:", metrics)
    
    # Save predictions
    test['p_lapse_3_m'] = probs
    test.to_csv('data/test_scored.csv', index=False)
    print("Saved data/test_scored.csv with predictions")
    
    # 6. Save Artifacts
    joblib.dump(model, 'churn_model_xgb.joblib')
    joblib.dump(encoder, 'feature_encoder.joblib')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 7. SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    print("Saved shap_summary.png")

    # Automated SHAP Analysis
    # shap_values is (n_samples, n_features). Mean absolute value per feature.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    top_3 = feature_importance.head(3)
    
    analysis_text = f"""
### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **{top_3.iloc[0]['feature']}**: Primary driver.
2. **{top_3.iloc[1]['feature']}**: Secondary driver.
3. **{top_3.iloc[2]['feature']}**: Tertiary driver.

See `shap_summary.png` for the full bar plot.
"""
    
    # Append to DISCUSSION.md
    with open('data/DISCUSSION.md', 'a', encoding='utf-8') as f:
        f.write(analysis_text)
    print("Appended SHAP analysis to data/DISCUSSION.md")

if __name__ == "__main__":
    train_xgboost_optuna()
