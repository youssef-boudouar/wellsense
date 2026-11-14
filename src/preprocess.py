import os
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

BASE = "/opt/airflow"


def preprocess():
    train = pd.read_csv(f"{BASE}/data/processed/train.csv")
    X = train.drop(columns=["Student_ID", "Mental_Health_Risk"])

    num_cols = ["Age", "Year_of_Study", "Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours", "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate"]
    ord_cols = ["Physical_Activity", "Diet_Quality", "Part_Time_Job", "Previous_Mental_Health_History"]
    ord_order = [["Low", "Moderate", "High"], ["Poor", "Average", "Good"], ["No", "Yes"], ["No", "Yes"]]

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        (make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(categories=ord_order)), ord_cols),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Gender"]),
    )

    X_prep = preprocessor.fit_transform(X)
    print("Preprocessed train:", X_prep.shape)

    joblib.dump(preprocessor, f"{BASE}/models/preprocessor.joblib")
    print("Preprocessor saved")