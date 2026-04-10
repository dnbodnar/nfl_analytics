# NFL EPA Prediction

A machine learning project analyzing Expected Points Added (EPA) in the NFL using four seasons of play-by-play data (2022-2025). The project compares 11 models across regression and classification tasks and uses SHAP values to interpret model decisions.

## Project Overview

EPA measures how much a single play changes a team's expected point total given the game situation. This project attempts to predict EPA from pre-snap context alone, revealing both the predictive power and the fundamental limits of situational football data.

## Structure

| Section | Description |
|---|---|
| 1. Setup and Data | Pull play-by-play data via nfl_data_py, store in SQLite, query with pandas SQL |
| 2. EDA | EPA distributions, team comparisons, Steelers spotlight |
| 3. Feature Engineering | Feature selection, target creation, sklearn Pipeline with ColumnTransformer |
| 4. Regression Models | Linear Regression, Ridge, Lasso, Random Forest, XGBoost |
| 5. Classification Models | Logistic Regression, Random Forest, XGBoost, SVM, KNN, Neural Network |
| 6. SHAP Values | Feature importance and model explainability on XGBoost |
| 7. Conclusions | Key findings, model comparison, limitations, and next steps |

## Key Findings

- Regression models produced near-zero R-squared scores, confirming that exact EPA is not predictable from pre-snap features alone
- Classification models reached ROC-AUC scores up to 0.61, outperforming the naive baseline but reflecting the inherent noise in individual play outcomes
- SHAP analysis identified yards to go and down as the most influential predictors, consistent with football intuition
- The core insight: individual play EPA is driven more by post-snap execution than pre-snap situation

## Stack

- Python 3.13, Jupyter Notebook
- Data: nfl_data_py, SQLite, SQLAlchemy
- ML: scikit-learn, XGBoost, TensorFlow/Keras
- Explainability: SHAP
- Visualization: Plotly, Matplotlib, Seaborn

## Setup

```bash
git clone https://github.com/dnbodnar/nfl_analytics.git
cd nfl_analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
