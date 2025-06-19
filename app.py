import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Compatibility fix for joblib with Streamlit Cloud
if __name__ != "__main__":
    import __main__
    __main__.__file__ = "app.py"

st.set_page_config(page_title="Flight & Customer Prediction", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'reset_triggered' not in st.session_state:
    st.session_state['reset_triggered'] = False

# Session Reset Button
if st.button("🔄 Reset Session"):
    st.session_state.clear()
    st.experimental_rerun()

st.title("✈️ Flight Price and Customer Satisfaction Prediction")

# Sidebar navigation
page = st.sidebar.selectbox("Select Project", ["Flight Price Prediction", "Customer Satisfaction Prediction"], key="selected_page")

le = LabelEncoder()

@st.cache_resource
def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model: `{path}`\n\nDetails: {e}")
        return None

# Tabs for both sections
if page == "Flight Price Prediction":
    with st.container():
        st.header("📈 Flight Price Prediction")
        df = pd.read_csv("data/Flight_Price.csv").drop_duplicates().dropna()

        df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.day
        df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.month
        df['Dep_Time_hour'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce').dt.hour
        df['Dep_Time_minute'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce').dt.minute
        df['Arrival_Time_hour'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce').dt.hour
        df['Arrival_Time_minute'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce').dt.minute

        def duration_to_mins(duration):
            h, m = 0, 0
            for part in duration.strip().split():
                if 'h' in part:
                    h = int(part.replace('h', ''))
                elif 'm' in part:
                    m = int(part.replace('m', ''))
            return h * 60 + m

        df['Duration_mins'] = df['Duration'].apply(duration_to_mins)
        for col in ['Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info']:
            df[col] = le.fit_transform(df[col])

        with st.expander("✍️ Input Flight Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                airline = st.selectbox("Airline", df['Airline'].unique(), key="airline")
                source = st.selectbox("Source", df['Source'].unique(), key="source")
                destination = st.selectbox("Destination", df['Destination'].unique(), key="destination")
                route = st.selectbox("Route", df['Route'].unique(), key="route")
            with col2:
                stops = st.selectbox("Total Stops", df['Total_Stops'].unique(), key="stops")
                day = st.slider("Journey Day", 1, 31, 15, key="journey_day")
                month = st.slider("Journey Month", 1, 12, 6, key="journey_month")
                dep_hour = st.slider("Departure Hour", 0, 23, 10, key="dep_hour")
                arr_hour = st.slider("Arrival Hour", 0, 23, 14, key="arr_hour")
                duration = st.slider("Duration (mins)", 30, 1000, 180, key="duration")

        input_data = pd.DataFrame({
            'Airline': [airline], 'Source': [source], 'Destination': [destination],
            'Route': [route], 'Total_Stops': [stops],
            'Journey_day': [day], 'Journey_month': [month],
            'Dep_Time_hour': [dep_hour], 'Dep_Time_minute': [0],
            'Arrival_Time_hour': [arr_hour], 'Arrival_Time_minute': [0],
            'Duration_mins': [duration]
        })

        model_name = st.selectbox("Select Model", ["linear_regression", "random_forest", "xgboost", "gradient_boosting", "knn_regressor"], key="flight_model")
        model_path = f"flight_price_prediction/models/{model_name}_compressed.pkl"
        model = load_model_safe(model_path)

        if model is not None:
            try:
                pred = model.predict(input_data)[0]
                st.success(f"💰 Predicted Flight Price: ₹{int(pred)}")
                st.download_button("📥 Download Prediction", str(int(pred)), file_name="flight_price.txt")
                st.session_state['history'].append({"project": "Flight", "prediction": int(pred)})
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

elif page == "Customer Satisfaction Prediction":
    with st.container():
        st.header("🧑‍✈️ Customer Satisfaction Prediction")
        original_df = pd.read_csv("data/Passenger_Satisfaction.csv").dropna()
        for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']:
            original_df[col] = le.fit_transform(original_df[col])

        scaler = load_model_safe("customer_satisfaction_prediction/models/scaler_compressed.pkl")

        st.subheader("✍️ Input Features")
        sample = original_df.drop(['id', 'satisfaction'], axis=1).iloc[[0]].copy()
        user_input = {}

        categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

        with st.form("customer_form"):
            for col in sample.columns:
                if col in categorical_columns:
                    categories = sorted(original_df[col].unique())
                    decoded = {i: val for i, val in enumerate(categories)}
                    encoded = {v: k for k, v in decoded.items()}
                    selected_label = st.selectbox(f"{col}", options=list(decoded.values()), key=f"input_{col}")
                    user_input[col] = selected_label
                    sample[col] = encoded[selected_label]
                else:
                    sample[col] = st.slider(f"{col}", min_value=0.0, max_value=100.0, value=float(sample[col].values[0]), step=1.0, key=f"input_{col}")
            submitted = st.form_submit_button("Submit")

        if submitted:
            input_scaled = scaler.transform(sample) if scaler else sample

            model_names = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "knn"]
            preds, probs = {}, {}

            for model_name in model_names:
                model_path = f"customer_satisfaction_prediction/models/{model_name}_compressed.pkl"
                model = load_model_safe(model_path)
                if model:
                    try:
                        preds[model_name] = model.predict(input_scaled)[0]
                        probs[model_name] = model.predict_proba(input_scaled)[0][1]
                    except Exception as e:
                        st.error(f"❌ Prediction failed for {model_name}: {e}")

            if preds:
                st.subheader("📊 Model Predictions")
                for name in preds:
                    st.write(f"🔹 **{name}**: {'✅ Satisfied' if preds[name] else '❌ Not Satisfied'} | Probability: {probs[name]:.2f}")

                final_vote = int(sum(preds.values()) >= 3)
                result = "Satisfied" if final_vote else "Not Satisfied"
                st.markdown(f"### 🧮 Final Prediction: **{'✅ Satisfied' if final_vote else '❌ Not Satisfied'}**")
                st.subheader("🔍 Top 3 Influential Features (Simulated)")
                for name in preds:
                    st.write(f"**{name}:** Flight Distance, Age, Inflight wifi service")
                st.download_button("📥 Download Satisfaction Prediction", result, file_name="satisfaction_result.txt")
                st.session_state['history'].append({"project": "Satisfaction", "prediction": result})

# 📦 Show Prediction History & CSV Download
if st.session_state['history']:
    st.markdown("---")
    st.subheader("📚 Prediction History")
    history_df = pd.DataFrame(st.session_state['history'])
    st.dataframe(history_df, use_container_width=True)
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download History as CSV", csv, file_name="prediction_history.csv")

st.caption("🔁 Powered by Streamlit | MLflow Ready")
