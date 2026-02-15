import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# -----------------------
# Load Model
# -----------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            font-weight:700;
            text-align:center;
        }
        .sub-text {
            text-align:center;
            color:gray;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown('<p class="main-title">📊 Student Performance Analytics System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Machine Learning Based Academic Score Prediction Dashboard</p>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Layout Columns
# -----------------------
left, right = st.columns([1, 1])

# -----------------------
# INPUT SECTION
# -----------------------
with left:
    st.subheader("📝 Student Input Panel")

    name = st.text_input("Student Name")

    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    internal_marks = st.slider("Internal Marks", 0, 100, 60)

    if st.button("🚀 Generate Prediction"):
        input_data = np.array([[study_hours, attendance, internal_marks]])
        prediction = model.predict(input_data)
        predicted_score = round(prediction[0], 2)

        # Grade Logic
        if predicted_score >= 85:
            grade = "A"
        elif predicted_score >= 70:
            grade = "B"
        elif predicted_score >= 50:
            grade = "C"
        else:
            grade = "Fail"

        # Save Prediction
        new_data = pd.DataFrame([[name, study_hours, attendance, internal_marks, predicted_score, grade]],
                                columns=["Name", "Study Hours", "Attendance", "Internal Marks", "Predicted Score", "Grade"])

        if os.path.exists("predictions.csv"):
            new_data.to_csv("predictions.csv", mode='a', header=False, index=False)
        else:
            new_data.to_csv("predictions.csv", index=False)

        st.success("Prediction Saved Successfully!")

        st.metric("Predicted Final Score", predicted_score)
        st.metric("Predicted Grade", grade)

# -----------------------
# ANALYTICS SECTION
# -----------------------
with right:
    st.subheader("📈 Performance Analytics")

    if os.path.exists("predictions.csv"):
        history = pd.read_csv("predictions.csv")

        if not history.empty:
            st.dataframe(history, use_container_width=True)

            st.markdown("### 📊 Score Distribution")
            st.bar_chart(history.set_index("Name")["Predicted Score"])

            st.markdown("### 📌 Summary Metrics")

            avg_score = round(history["Predicted Score"].mean(), 2)
            max_score = history["Predicted Score"].max()
            min_score = history["Predicted Score"].min()

            col1, col2, col3 = st.columns(3)
