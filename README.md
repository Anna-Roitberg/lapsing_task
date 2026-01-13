# Insurance Policy Lapse Prediction & Strategy Generation

A machine learning system that predicts insurance policy lapses and generates personalized retention strategies using RAG (Retrieval-Augmented Generation).

## Overview

This project implements an end-to-end pipeline for:
1. **Lapse Prediction**: XGBoost classifier to predict 3-month lapse probability
2. **Retention Strategy**: RAG-based personalized retention plans for at-risk customers
3. **Conversion Planning**: Targeted 3-step conversion plans for new leads

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
python run.py
```

This orchestrates:
1. Model training with Optuna hyperparameter tuning
2. Strategy generation for existing customers
3. Conversion plan generation for new leads

## Project Structure

### Core Scripts

#### `train_model.py`
Trains the XGBoost lapse prediction model with:
- **Feature Engineering**: Bins `age`, `premium`, and `tenure_m` into categorical ranges
- **Hyperparameter Tuning**: Optuna optimization (30 trials)
- **Evaluation**: AUC-PR, Precision@K metrics
- **Explainability**: SHAP analysis for feature importance
- **Outputs**: 
  - `churn_model_xgb.joblib` (trained model)
  - `feature_encoder.joblib` (fitted encoders)
  - `metrics.json`, `shap_summary.png`

#### `generate_strategy.py`
Generates retention strategies for at-risk customers:
- **Input**: `data/three_test_customers_high_med_low_risk.csv` (or test data)
- **Process**:
  1. Loads model and encoder
  2. Predicts lapse probability
  3. Retrieves relevant playbook snippets (RAG from `rag_docs/lapse/`)
  4. Constructs retention prompt with context + citations
- **Output**: Console display of risk tier, retrieved docs, and mock strategy plan

#### `generate_conversion_plan.py`
Creates conversion plans for new insurance leads:
- **Input**: `data/three_lead_profiles_small.csv`
- **Process**:
  1. Infers customer context (`channel`, `needs`, `objections`) from demographics
  2. Retrieves objection handling advice (RAG from `rag_docs/leads/`)
  3. Generates 3-step conversion plan
- **Output**: JSON-structured plan with full-text citations

#### `run.py`
Orchestrates the full workflow sequentially.

### Supporting Modules

- **`retrieval_system.py`**: TF-IDF based RAG implementation
- **`strategy_contract.py`**: Data structures for retention context
- **`strategy_prompt.py`**: Prompt templates for retention LLM
- **`conversion_contract.py`**: Data structures for conversion context  
- **`conversion_prompt.py`**: Prompt templates for conversion LLM
- **`split_data.py`**: Utility to split GPT-generated data by `split` column

## Data

### Input Files
- `data/train_gpt.csv`, `data/val_gpt.csv`, `data/test_gpt.csv`: Training splits
- `data/lead_policies_3.csv`: Sample at-risk policies for strategy generation
- `data/lead_customers_3.csv`: Sample new leads for conversion planning

### RAG Knowledge Base
- `rag_docs/lapse/`: Retention playbooks (e.g., agent outreach, billing)
- `rag_docs/leads/`: Conversion guides (e.g., objection handling, value props)

## Feature Engineering

The model uses categorical binning for interpretability:

| Feature | Bins |
|---------|------|
| **age** | 18-30, 31-45, 46-60, 61+ |
| **premium** | <100, 100-150, 150-200, 200-300, 300+ |
| **tenure_m** | 0-6m, 6-12m, 12-24m, 24-48m, 48m+ |
| **coverage** | <10k, 10k-25k, 25k-50k, 50k-100k, 100k+ |

## Model Performance

Evaluated on:
- **AUC-PR**: Average precision (lapse is rare event)
- **Precision@1%**: Precision in top 1% scored policies
- **Precision@5%**: Precision in top 5% scored policies

See `metrics.json` after training.

## Example Usage

### Predict and Generate Strategy
```bash
python generate_strategy.py
```

### Generate Conversion Plans
```bash
python generate_conversion_plan.py
```

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies
