import pandas as pd
import joblib
from app.rag.retail_ai_agent import retail_rag_response


# =========================================
# LOAD MODELS
# =========================================

forecast_model = joblib.load(
    "models/best_forecasting_model.pkl"
)

anomaly_model = joblib.load(
    "models/anomaly_detection_model.pkl"
)


# =========================================
# FORECAST AGENT
# =========================================

def forecast_agent():

    input_data = pd.DataFrame([{

        "region": "West",
        "segment": "Corporate",
        "shipping_mode": "First Class",
        "category": "Electronics",
        "product_name": "Laptop",
        "quantity": 5,
        "discount": 0.05,
        "year": 2025,
        "month": 11,
        "day": 15,
        "weekday": 5
    }])

    prediction = forecast_model.predict(
        input_data
    )[0]

    return f"Predicted Sales: {round(prediction, 2)}"


# =========================================
# ANOMALY AGENT
# =========================================

def anomaly_agent():

    input_data = pd.DataFrame([{

        "sales": 150000,
        "quantity": 2,
        "discount": 0.01,
        "profit": 100
    }])

    prediction = anomaly_model.predict(
        input_data
    )[0]

    if prediction == -1:

        return "Anomaly Detected"

    else:

        return "Normal Transaction"


# =========================================
# RETRIEVAL AGENT
# =========================================

def retrieval_agent(query):
    return retail_rag_response(query)

    


# =========================================
# COORDINATOR AGENT
# =========================================

def coordinator_agent(user_query):

    query = user_query.lower()

    if "forecast" in query or "sales" in query:

        return forecast_agent()

    elif "anomaly" in query or "fraud" in query:

        return anomaly_agent()

    else:

        return retrieval_agent(user_query)