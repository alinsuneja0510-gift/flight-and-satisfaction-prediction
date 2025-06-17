import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Fix for joblib pickle compatibility in Streamlit Cloud
if __name__ != "__main__":
    import __main__
    __main__.__file__ = "app.py"

st.set_page_config(page_title="Flight & Customer Prediction", layout="wide")
st.title("✈️ Flight Price and Customer Satisfaction Prediction")

# Sidebar for selecting project
page = st.sidebar.selectbox("Select Project", ["Flight Price Prediction", "Customer Satisfaction Prediction"])

le = LabelEncoder()

# ✅ Safe model loader
@st.cache_resource
def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model: `{path}`\n\nDetails: {e}")
        return None

# ---------------------------
# 🔹 Flight Price Prediction
# ---------------------------
if page == "Flight Price Prediction":
    st.header("📈 Flight Price Prediction")
    df = pd.read_csv("data/Flight_Price.csv").drop_duplicates().dropna()

    # Feature engineering
    df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.day
    df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.month
    for col in ['Dep_Time', 'Arrival_Time']:
        df[col + '_hour'] = pd.to_datetime(df[col].str.split().str[0], format='%H:%M', errors='coerce').dt.hour
        df[col + '_minute'] = pd.to_datetime(df[col].str.split().str[0], format='%H:%M', errors='coerce').dt.minute

    def duration_to_mins(duration):
        h, m = 0, 0
        for part in duration.strip().split():
            if 'h' in part: h = int(part.replace('h', ''))
            elif 'm' in part: m = int(part.replace('m', ''))
        return h * 60 + m

    df['Duration_mins'] = df['Duration'].apply(duration_to_mins)
    for col in ['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info']:
        df[col] = le.fit_transform(df[col])

    st.subheader("Input Features")
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Airline", df['Airline'].unique())
        source = st.selectbox("Source", df['Source'].unique())
        destination = st.selectbox("Destination", df['Destination'].unique())
        route = st.selectbox("Route", df['Route'].unique())
    with col2:
        stops = st.selectbox("Total Stops", df['Total_Stops'].unique())
        day = st.slider("Journey Day", 1, 31, 15)
        month = st.slider("Journey Month", 1, 12, 6)
        dep_hour = st.slider("Departure Hour", 0, 23, 10)
        arr_hour = st.slider("Arrival Hour", 0, 23, 14)
        duration = st.slider("Duration (mins)", 30, 1000, 180)

    input_data = pd.DataFrame({
        'Airline': [airline], 'Source': [source], 'Destination': [destination], 'Route': [route],
        'Total_Stops': [stops], 'Journey_day': [day], 'Journey_month': [month],
        'Dep_Time_hour': [dep_hour], 'Dep_Time_minute': [0],
        'Arrival_Time_hour': [arr_hour], 'Arrival_Time_minute': [0], 'Duration_mins': [duration]
    })

    model_name = st.selectbox("Select Model", ["linear_regression", "random_forest", "xgboost", "gradient_boosting", "knn"])
    model = load_model_safe(f"flight_price_prediction/models/{model_name}.pkl")

    if model:
        pred = model.predict(input_data)[0]
        st.success(f"💰 Predicted Flight Price: ₹{int(pred)}")
        st.download_button("📥 Download Prediction", str(int(pred)), file_name="flight_price.txt")

# ---------------------------
# 🔹 Customer Satisfaction
# ---------------------------
elif page == "Customer Satisfaction Prediction":
    st.header("🧑‍✈️ Customer Satisfaction Prediction")
    df = pd.read_csv("data/Passenger_Satisfaction.csv").dropna()

    for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']:
        df[col] = le.fit_transform(df[col])

    st.subheader("Input Features")
    gender = st.selectbox("Gender", df['Gender'].unique())
    age = st.slider("Age", 10, 80, 35)
    travel_type = st.selectbox("Type of Travel", df['Type of Travel'].unique())
    travel_class = st.selectbox("Class", df['Class'].unique())
    flight_distance = st.slider("Flight Distance", 50, 5000, 800)
    inflight_wifi = st.slider("Inflight Wifi", 0, 5, 3)

    sample = df.drop(['id', 'satisfaction'], axis=1).iloc[[0]].copy()
    sample['Gender'] = gender
    sample['Age'] = age
    sample['Type of Travel'] = travel_type
    sample['Class'] = travel_class
    sample['Flight Distance'] = flight_distance
    sample['Inflight wifi service'] = inflight_wifi

    model_names = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "knn"]
    preds = {}
    probs = {}

    for model_name in model_names:
        model = load_model_safe(f"customer_satisfaction_prediction/models/{model_name}.pkl")
        if model:
            pred = model.predict(sample)[0]
            prob = model.predict_proba(sample)[0][1]
            preds[model_name] = pred
            probs[model_name] = prob

    st.subheader("📊 Model Predictions")
    for name in model_names:
        if name in preds:
            st.write(f"🔹 {name}: {'✅ Satisfied' if preds[name] else '❌ Not Satisfied'} | Probability: {probs[name]:.2f}")

    if preds:
        final_vote = int(sum(preds.values()) >= 3)
        st.markdown(f"### 🧮 Final Prediction: {'✅ Satisfied' if final_vote else '❌ Not Satisfied'}")
        st.subheader("🔍 Top 3 Influential Features (Simulated)")
        for name in preds:
            st.write(f"**{name}:** Flight Distance, Age, Inflight wifi service")
        st.download_button("📥 Download Satisfaction Prediction", "Satisfied" if final_vote else "Not Satisfied", file_name="satisfaction_result.txt")

st.caption("🔁 Powered by Streamlit | MLflow Ready")
