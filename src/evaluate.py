import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = "/opt/airflow"


def evaluate():
    test_df = pd.read_csv(f"{BASE}/data/processed/test.csv")
    X_test = test_df.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y_test = test_df["Mental_Health_Risk"]

    model = joblib.load(f"{BASE}/models/wellsense_model_airflow.joblib")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]).tolist(),
    }
    print(metrics)

    with open(f"{BASE}/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved")