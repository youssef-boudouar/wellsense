# src/prepare.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "student_mental_health.csv"
PROCESSED = ROOT / "data" / "processed"

TARGET = "Mental_Health_Risk"


def prepare_data():
    df = pd.read_csv(RAW_DATA)

    inconsistent = df["Social_Media_Hours"] > df["Screen_Time_Hours"]
    df = df[~inconsistent]

    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=77
    )

    PROCESSED.mkdir()
    train.to_csv(PROCESSED / "train.csv", index=False) # save without unnamed : 0 pandas default index
    test.to_csv(PROCESSED / "test.csv", index=False)
    print(f"train: {train.shape}, test: {test.shape}")