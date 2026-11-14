# app/app.py
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

st.title("WellSense")

model = joblib.load("models/wellsense_model.joblib")

with open("reports/metrics.json") as f:
    metrics = json.load(f)


st.header("Model performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", round(metrics["accuracy"], 3))
col2.metric("Macro F1", round(metrics["f1_macro"], 3))

st.subheader("Confusion matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]),
                       display_labels=["Low", "Medium", "High"]).plot(ax=ax)
st.pyplot(fig)


st.header("Predict for a new student")

with st.form("student_form"):
    age = st.number_input("Age", 17, 30, 21)
    year = st.number_input("Year of study", 1, 5, 2)
    sleep = st.number_input("Sleep hours", 0.0, 12.0, 7.0)
    study = st.number_input("Study hours per week", 0.0, 60.0, 20.0)
    gpa = st.number_input("GPA", 0.0, 4.0, 2.5)
    social = st.number_input("Social media hours", 0.0, 16.0, 6.0)
    screen = st.number_input("Screen time hours", 0.0, 16.0, 8.0)
    finance = st.number_input("Financial stress (1-10)", 1.0, 10.0, 5.0)
    family = st.number_input("Family support (1-10)", 1.0, 10.0, 6.0)
    attendance = st.number_input("Attendance rate (%)", 0.0, 100.0, 85.0)
    gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
    activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
    diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
    job = st.selectbox("Part-time job", ["No", "Yes"])
    history = st.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    if social > screen:
        st.error("Social media hours cannot be higher than screen time.")
    else:
        new_student = pd.DataFrame([{
            "Age": age, "Gender": gender, "Year_of_Study": year, "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study, "GPA": gpa, "Social_Media_Hours": social,
            "Screen_Time_Hours": screen, "Physical_Activity": activity,
            "Financial_Stress": finance, "Family_Support": family,
            "Attendance_Rate": attendance, "Part_Time_Job": job,
            "Diet_Quality": diet, "Previous_Mental_Health_History": history,
        }])
        new_student = new_student[model.feature_names_in_]

        prediction = model.predict(new_student)[0]
        proba = model.predict_proba(new_student)[0]

        st.subheader(f"Estimated risk level: {prediction}")
        st.bar_chart(pd.DataFrame({"probability": proba}, index=model.classes_))