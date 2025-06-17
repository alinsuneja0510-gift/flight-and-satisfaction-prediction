# ✈️ Flight Price & Customer Satisfaction Prediction App

## 📌 Project Overview

This Streamlit web app combines two end-to-end machine learning projects:

1. **Flight Price Prediction (Regression)** – Predicts flight ticket prices based on features like airline, route, duration, etc.
2. **Customer Satisfaction Prediction (Classification)** – Predicts whether a passenger is satisfied based on service ratings, travel class, flight distance, and more.

Both models are trained, evaluated, tracked using **MLflow**, and deployed through **Streamlit Cloud**.

---

## 📁 Project Structure

```
Project 3/
│
├── app.py                                 # Main Streamlit app with page navigation
├── requirements.txt                       # Required packages for deployment
├── README.md                              # Project documentation
│
├── flight_price_prediction/               # Scripts & models for flight price prediction
├── customer_satisfaction_prediction/      # Scripts & models for satisfaction prediction
├── preprocessing/                         # Common preprocessing utilities
└── data/                                  # Raw CSV files (Flight_Price.csv, Passenger_Satisfaction.csv)
```

---

## 🧠 Skills Demonstrated

- Python, Pandas, NumPy
- Machine Learning: Regression & Classification
- MLflow Experiment Tracking
- Streamlit App Development
- Data Cleaning & Feature Engineering

---

## 🚀 Streamlit App Features

- 📊 **Visualizations** of flight prices and satisfaction trends
- 🧮 Predict flight prices based on user inputs
- 🎯 Predict satisfaction levels of passengers
- 🔎 View feature importance and SHAP values
- 📤 Export predictions

---

## 🧪 MLflow Tracking

Each model is logged with:

- Parameters (e.g. model type, hyperparameters)
- Metrics (RMSE, Accuracy, F1-score, etc.)
- Artifacts (models, plots)
- Tracked using `mlflow.set_experiment()`

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Scikit-learn**
- **XGBoost**
- **MLflow**
- **Pandas / NumPy / Matplotlib / Seaborn**

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/your-username/flight-satisfaction-streamlit-app.git
cd flight-satisfaction-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔗 Deployed App

🌐 [Live Streamlit App](https://share.streamlit.io/your-username/flight-satisfaction-streamlit-app)

---

## 🧾 Author

- **Name:** Alin Suneja
- **GitHub:** [alinsuneja0510-gift](https://github.com/alinsuneja0510-gift)
- **Date:** 14 June 2025

---