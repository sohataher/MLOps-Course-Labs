# Bank Churn Prediction with MLflow

A simple ML workflow for predicting bank customer churn using MLflow to track experiments and models.

## Features

- Data preprocessing (scaling, encoding, rebalancing)
- Model training: Logistic Regression, Random Forest, XGBoost
- Metrics: accuracy, precision, recall, F1
- Logs: models, confusion matrix, transformer
- Model registry integration

## Output

- Logged in MLflow UI under `Churn Prediction` experiment
- Registered models:
  - `LogisticRegression`
  - `RandomForestChurnModel`
  - `XGBoostChurnModel`


## Artifacts

- `column_transformer.pkl`