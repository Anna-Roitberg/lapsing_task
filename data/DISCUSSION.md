# Data Documentation

## Dataset Overview
The dataset contains insurance policy records tracked monthly over the year 2023.
Target variable: `lapse_next_3m` (Binary: 1 if policy lapses in next 3 months).

## Features
*   **policy_id**: Unique identifier.
*   **month**: Reporting month.
*   **age**: Policyholder age.
*   **tenure_m**: Months since policy start.
*   **premium**: Monthly premium amount.
*   **coverage**: Total insured amount.
*   **region**: Geographic region.
*   **has_agent**: Boolean, if customer has an assigned agent.
*   **is_smoker**: Boolean.
*   **dependents**: Number of dependents.

## Known Issues

### 1. Leakage Trap: `post_event_notice_sent`
> [!WARNING]
> The feature `post_event_notice_sent` is highly correlated with the target because it is a **consequence** of the lapse event (a notice sent *after* a missed payment or lapse trigger).
> **Action**: This column MUST be excluded from the training feature set to avoid target leakage.

### 2. Concept Drift
> [!NOTE]
> Lapse rates show a distinct increase starting from **July 2023** due to external economic factors simulated in the data. Models trained only on H1 2023 might underestimate risk in H2.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **age**: Primary driver.
2. **premium**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **age**: Primary driver.
2. **premium**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **age**: Primary driver.
2. **premium**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **age**: Primary driver.
2. **coverage**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **age**: Primary driver.
2. **has_agent**: Secondary driver.
3. **coverage**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **coverage**: Primary driver.
2. **age**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **coverage**: Primary driver.
2. **age**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.

### SHAP Analysis
The global feature importance (mean absolute SHAP value) indicates the top drivers of lapse risk:
1. **coverage**: Primary driver.
2. **age**: Secondary driver.
3. **has_agent**: Tertiary driver.

See `shap_summary.png` for the full bar plot.
