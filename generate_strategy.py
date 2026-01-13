import pandas as pd
import joblib
import json
import os
import random
from sklearn.preprocessing import OrdinalEncoder

# Import our components
from retrieval_system import MinimalRAG
from strategy_contract import CustomerContext
from strategy_prompt import StrategyPromptBuilder

def load_system():
    print("Loading XGBoost Model...")
    model = joblib.load('churn_model_xgb.joblib')
    
    print("Loading Feature Encoder...")
    encoder = joblib.load('feature_encoder.joblib')
    
    print("Initializing RAG System...")
    rag = MinimalRAG()
    
    return model, encoder, rag

def prepare_and_score_data(file_path, model, encoder):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame()
        
    print(f"Loading and scoring data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Features from train_model.py
    feature_cols = ['age', 'tenure_m', 'premium', 'coverage', 'region', 'has_agent', 'is_smoker', 'dependents']
    
    # Check if all required columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing} in {file_path}")
        return pd.DataFrame()
    
    # Create feature matrix X
    X = df[feature_cols].copy()
    
    # Bin features (same as training)
    # Age binning
    X['age'] = pd.cut(X['age'], 
                     bins=[0, 30, 45, 60, 150], 
                     labels=['18-30', '31-45', '46-60', '61+'],
                     right=True)
    
    # Premium binning
    X['premium'] = pd.cut(X['premium'], 
                         bins=[0, 100, 150, 200, 300, 10000], 
                         labels=['<100', '100-150', '150-200', '200-300', '300+'],
                         right=True)
    
    # Tenure binning
    X['tenure_m'] = pd.cut(X['tenure_m'], 
                          bins=[-1, 6, 12, 24, 48, 10000], 
                          labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+'],
                          right=True)
    
    # Apply Encoding for all categorical features
    X[['region', 'age', 'premium', 'tenure_m']] = encoder.transform(X[['region', 'age', 'premium', 'tenure_m']])

        
    # Predict
    probs = model.predict_proba(X)[:, 1]
    
    # Attach to dataframe
    df['p_lapse_3_m'] = probs
    return df

def run_strategy_pipeline(policy_row, rag):
    policy_id = policy_row.get('policy_id', 'Unknown')
    print(f"\n{'='*60}")
    print(f"Processing Policy: {policy_id}")
    print(f"{'='*60}")
    
    p_lapse = policy_row['p_lapse_3_m']
    
    # Defaulting optional fields if they don't exist in the CSV (like leads file)
    payment_status = str(policy_row.get('payment_status', 'Paid'))
    if 'payment_status' not in policy_row:
        # Simple heuristic for demo: if high risk, maybe late?
        payment_status = 'Late' if p_lapse > 0.6 else 'Paid'
        
    context = CustomerContext(
        policy_id=policy_id,
        month=str(policy_row.get('month', '2023-12')),
        policy_age=int(policy_row.get('age', 12)),
        premium_amount=float(policy_row.get('premium', 100.0)),
        payment_status=payment_status,
        customer_calls=int(policy_row.get('call_count', 0)),
        claim_count=int(policy_row.get('claim_count', 0)),
        p_lapse=p_lapse,
        risk_tier=StrategyPromptBuilder.determine_risk_tier(p_lapse)
    )
    
    print(f"Risk Profile: {context.risk_tier} (Prob: {p_lapse:.4f})")
    
    # 2. Generate Retrieval Query
    query = context.to_retrieval_query()
    print(f"Generated Query: \"{query}\"")
    
    # 3. Retrieve Config
    results = rag.retrieve(query, k=3)
    print(f"Retrieved {len(results)} snippets.")
    for r in results:
        print(f"  - [{r['score']:.2f}] {r['source']}")
        
    # 4. Build Prompt
    messages = StrategyPromptBuilder.build_messages(context, results)
    
    print("\n--- LLM Prompt (User Message Snippet) ---")
    print(messages[1]['content'][:500] + "...")
    print("-----------------------------------------")
    
    # 5. Mock LLM Response
    print("\n[MOCK] LLM Generation:")
    mock_response = {
        "risk_tier": context.risk_tier,
        "primary_driver": "Price Sensitivity" if context.premium_amount > 150 else "Engagement Drop",
        "actions": [
            {
                "action_name": "Proactive Rate Review",
                "reasoning": "Customer is watchlist tier with high premium.",
                "expected_impact": "High",
                "cost_effort": "Medium"
            }
        ],
        "message_templates": [
            {
                "channel": "Email",
                "content": f"Hi, we noticed your premium is ${context.premium_amount}. Let's discuss options..."
            }
        ],
        "metrics_to_track": ["Renewal Conservation"]
    }
    print(json.dumps(mock_response, indent=2))

def main():
    try:
        model, encoder, rag = load_system()

        target_file = 'data/test_lapse_customers_3.csv'

        print(f"Targeting file: {target_file}")
        
        scored_df = prepare_and_score_data(target_file, model, encoder)
        
        if scored_df.empty:
            print("No valid policies to process.")
            return

        # Select a few interesting cases to display
        # 1. Highest Risk
        high_risk = scored_df.sort_values('p_lapse_3_m', ascending=False).head(1)
        
        # 2. Some Moderate Risk (middle of the pack)
        mid_risk = scored_df[(scored_df['p_lapse_3_m'] > 0.3) & (scored_df['p_lapse_3_m'] < 0.7)].head(1)
        
        # 3. Low Risk
        low_risk = scored_df.sort_values('p_lapse_3_m', ascending=True).head(1)
        
        demo_set = pd.concat([high_risk, mid_risk, low_risk])
        
        for _, row in demo_set.iterrows():
            run_strategy_pipeline(row, rag)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
