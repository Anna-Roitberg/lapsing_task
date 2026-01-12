# DISCUSSION

## Dataset
- Panel size: ~2k policies × 12 months (2023-01..2023-12), one row per (policy_id, month).
- Features:
  policy_id, month, age, tenure_m, premium, coverage, region, has_agent, is_smoker, dependents
- Target:
  **lapse_next_3m** = 1 if the policy lapses within a 3-month window starting at the current month:
  lapse_month ∈ [month, month+2]. Otherwise 0.

## Drift (simple concept drift)
A drift is introduced starting **2023-07**:
- Premiums slightly increase (e.g., macro pricing adjustment).
- Lapse hazard increases (e.g., affordability shock).

Observed synthetic label rate:
- Overall: 0.073
- Pre-drift (< 2023-07): 0.074
- Post-drift (>= 2023-07): 0.072

## Leakage trap feature (DO NOT USE IN MODEL)
This dataset includes a deliberate leakage feature:

- **post_event_notice_sent**

It is computed using future knowledge (it is nearly the label with small noise). In real life, such a signal would only exist **after** the lapse event is known/recorded. Including it in training will cause unrealistically high performance.

✅ Exclude all `post_event_*` features from modeling.

## Time split (strict)
Strict time-based split by month:
- Train: 2023-01 .. 2023-08
- Val:   2023-09 .. 2023-10
- Test:  2023-11 .. 2023-12

This respects temporal ordering (no future months in training).

## Notes
- Labeling uses an internal simulation that extends to 2024-03 to allow the "next 3 months" label for late 2023 months,
  while still outputting only 12 months of feature rows.

## Model (XGBoost)
One fast XGBoost classifier is trained on the strict temporal split (`train` / `val`).
- **Algorithm**: XGBoost Classifier (Optuna tuned).
- **Target**: `lapse_next_3m`.
- **Metrics**: AUC-PR (Primary), Precision@1%, Precision@5%.
- **Interpretation**: Global feature importance via SHAP (see `shap_summary.png`).


### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **tenure_m**: Primary driver.
2. **age**: Secondary driver.
3. **coverage**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **tenure_m**: Primary driver.
2. **age**: Secondary driver.
3. **coverage**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **tenure_m**: Primary driver.
2. **coverage**: Secondary driver.
3. **age**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **tenure_m**: Primary driver.
2. **coverage**: Secondary driver.
3. **age**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **tenure_m**: Primary driver.
2. **coverage**: Secondary driver.
3. **age**: Tertiary driver.

See `shap_summary.png` for the full bar plot.
