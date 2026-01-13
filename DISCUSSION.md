# Synthetic Policy Lapse Dataset (2k policies × 12 months)

This dataset is **synthetic** but designed to look realistic for training a time-aware lapse prediction model.

## Data shape
- ~2000 policies × 12 monthly records (2023-01 … 2023-12)
- One row = (policy_id, month) snapshot
- Target: `lapse_next_3m` = 1 if the policy lapses during the **next 3 months** after `month`

## Feature semantics (intuitive)
- `age`: policyholder age (18–85)
- `tenure_m`: months since policy start (higher tenure → more stable → lower lapse)
- `coverage`: insured amount (log-normal, heavy tail)
- `premium`: monthly price, correlated with coverage and risk factors
- `region`: categorical (coastal/central/north/south) with small socioeconomic differences
- `has_agent`: agent involvement (reduces lapse)
- `is_smoker`: small increase in lapse (higher premium / risk profile)
- `dependents`: small increase in lapse (household budget pressure)

## Lapse probability design (real-like distribution)
The target is generated from a logistic model with noise:
- **Younger** customers are more likely to lapse
- **Higher premium burden** (premium / income proxy) increases lapse
- **Low tenure** increases lapse (early churn)
- No agent slightly increases lapse
- Small region and smoker/dependents effects
- Policy-level random intercept adds unobserved heterogeneity

## Concept drift
A simple drift starts **2023-07** (inclusive) to emulate a macro shock:
- Global uplift in lapse probability after 2023-07
- Stronger effect for **more expensive** policies (interaction with premium)

## Leakage trap feature (DO NOT USE FOR MODELING)
`post_event_notice_sent` is a **deliberate leakage trap**:
- It represents an operational action that typically happens **after** a missed payment / imminent lapse risk.
- In the data it is strongly correlated with `lapse_next_3m`, so including it will create **leakage** and unrealistically high performance.
- **Exclude this feature** from training/evaluation.

## Time-based split (strict)
Column `split` is derived only from `month` (no leakage):
- Train: 2023-01 … 2023-08
- Val:   2023-09 … 2023-10
- Test:  2023-11 … 2023-12

This enforces a strict **forward-chaining** setup and naturally contains the drift in later periods.



## Known Issues

### 1. Leakage Trap: `post_event_notice_sent`

> [!WARNING] The feature `post_event_notice_sent` is highly correlated with the target because it is a **consequence** of the lapse event (a notice sent *after* a missed payment or lapse trigger). **Action**: This column MUST be excluded from the training feature set to avoid target leakage.

### 2. Concept Drift

> [!NOTE] Lapse rates show a distinct increase starting from **July 2023** due to external economic factors simulated in the data. Models trained only on H1 2023 might underestimate risk in H2.

### SHAP Analysis

The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:

1. **coverage**: Primary driver.
2. **age**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.