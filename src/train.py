# src/train.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

BASE = "/opt/airflow"


def train():

    train_df = pd.read_csv(f"{BASE}/data/processed/train.csv")
    X = train_df.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y = train_df["Mental_Health_Risk"]

    preprocessor = joblib.load(f"{BASE}/models/preprocessor.joblib")

    model = make_pipeline(preprocessor, LogisticRegression(C=1, max_iter=1000))
    model.fit(X, y)

    joblib.dump(model, f"{BASE}/models/wellsense_model_airflow.joblib")