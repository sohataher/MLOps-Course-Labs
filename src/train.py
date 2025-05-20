"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import joblib

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    joblib.dump(col_transf, "column_transformer.pkl")
    mlflow.log_artifact("column_transformer.pkl", artifact_path="preprocessing")

    return col_transf, X_train, X_test, y_train, y_test

from mlflow.models.signature import infer_signature

def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))

    # Log model
    # with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="LogisticRegression"
    )

    ### Log the data
    mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
    mlflow.log_metric("train_accuracy", log_reg.score(X_train, y_train))


    return log_reg

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    signature = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="RandomForestChurnModel"
    )

    mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100})
    mlflow.log_metric("train_accuracy", rf.score(X_train, y_train))

    return rf


from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)

    signature = infer_signature(X_train, xgb.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=xgb,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="XGBoostChurnModel"
    )

    mlflow.log_params({"model_type": "XGBoost"})
    mlflow.log_metric("train_accuracy", xgb.score(X_train, y_train))

    return xgb


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    mlflow.set_experiment("Churn Prediction")

    for model_name, train_func in {
        "LogisticRegression": train,
        "RandomForest": train_random_forest,
        "XGBoost": train_xgboost,  
    }.items():
        with mlflow.start_run():
            df = pd.read_csv("dataset/Churn_Modelling.csv")
            col_transf, X_train, X_test, y_train, y_test = preprocess(df)

            mlflow.set_tag("model_name", model_name)
            model = train_func(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        
            conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
            conf_mat_disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_mat, display_labels=model.classes_
            )
            conf_mat_disp.plot()
            
            # Log the image as an artifact in MLflow
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
            
            plt.show()


if __name__ == "__main__":
    main()
