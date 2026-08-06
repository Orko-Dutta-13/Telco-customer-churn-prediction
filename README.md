# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription using machine learning.

## Dataset
IBM Telco Customer Churn dataset — 7,032 customers, 21 features

## Models Used
- Logistic Regression
- Random Forest  
- XGBoost (final model)

## Results
| Model | Accuracy | AUC Score | Churn Recall |
|-------|----------|-----------|--------------|
| Logistic Regression | 72.6% | ~0.83 | 80% |
| Random Forest | 76.1% | ~0.83 | 65% |
| XGBoost | 75.3% | ~0.83 | 69% |

## Key Findings
- Month-to-month contract customers churn the most
- New customers (0–10 months tenure) are the highest risk group
- Fiber optic customers churn despite paying the most
- High monthly charges ($70+) strongly predict churn

## Tools Used
Python, Jupyter Notebook, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
