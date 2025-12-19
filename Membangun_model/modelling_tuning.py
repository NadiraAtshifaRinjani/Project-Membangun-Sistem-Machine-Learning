import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Load data
df = pd.read_csv("data_preprocessed.csv")

X = df.drop(columns=["price", "price_log"])
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42
}

with mlflow.start_run(run_name="RandomForest_Tuning"):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Manual logging
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Artefak 1: model
    joblib.dump(model, "rf_model.pkl")
    mlflow.log_artifact("rf_model.pkl")

    # Artefak 2: feature importance plot
    importances = model.feature_importances_
    plt.figure(figsize=(10,4))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
