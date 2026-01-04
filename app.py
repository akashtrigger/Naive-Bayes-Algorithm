import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Naive Bayes Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Naive Bayes Prediction App")
st.write("Predict whether a customer will **Purchase** or **Not Purchase**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_model.pkl")

model = load_model()

# ---------------- USER INPUT ----------------
st.subheader("Enter Customer Details")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Salary", min_value=1000, max_value=200000, value=40000)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Customer WILL PURCHASE")
        st.balloons()
    else:
        st.error("❌ Customer WILL NOT PURCHASE")
    