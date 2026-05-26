from logs.logger import logger

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from app.agents.coordinator_agent import coordinator_agent

# =========================================
# MONGODB IMPORTS
# =========================================

from database.mongodb import (
    sales_collection,
    anomaly_collection,
    chat_collection
)

# =========================================
# CREATE APP
# =========================================

app = FastAPI(
    title="Smart Retail AI Platform"
)

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
# REQUEST SCHEMAS
# =========================================

class ForecastRequest(BaseModel):

    region: str
    segment: str
    shipping_mode: str
    category: str
    product_name: str
    quantity: int
    discount: float
    year: int
    month: int
    day: int
    weekday: int


class AnomalyRequest(BaseModel):

    sales: float
    quantity: int
    discount: float
    profit: float


class ChatRequest(BaseModel):

    query: str


# =========================================
# HOME API
# =========================================

@app.get("/")
def home():

    logger.info("Home API accessed")

    return {
        "message": "Smart Retail AI Platform Running"
    }


# =========================================
# SALES PREDICTION API
# =========================================

@app.post("/predict-sales")
def predict_sales(request: ForecastRequest):

    try:

        input_data = pd.DataFrame([{

            "region": request.region,
            "segment": request.segment,
            "shipping_mode": request.shipping_mode,
            "category": request.category,
            "product_name": request.product_name,
            "quantity": request.quantity,
            "discount": request.discount,
            "year": request.year,
            "month": request.month,
            "day": request.day,
            "weekday": request.weekday

        }])

        prediction = forecast_model.predict(
            input_data
        )[0]

        result = round(
            float(prediction),
            2
        )

        logger.info(
            f"Sales prediction generated for {request.product_name}"
        )

        # =========================================
        # SAVE TO MONGODB
        # =========================================

        sales_collection.insert_one({

            "region": request.region,
            "segment": request.segment,
            "shipping_mode": request.shipping_mode,
            "category": request.category,
            "product_name": request.product_name,
            "quantity": request.quantity,
            "discount": request.discount,
            "year": request.year,
            "month": request.month,
            "day": request.day,
            "weekday": request.weekday,
            "predicted_sales": result

        })

        return {

            "predicted_sales": result
        }

    except Exception as e:

        logger.error(f"Prediction API Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# ANOMALY DETECTION API
# =========================================

@app.post("/detect-anomaly")
def detect_anomaly(request: AnomalyRequest):

    try:

        input_data = pd.DataFrame([{

            "sales": request.sales,
            "quantity": request.quantity,
            "discount": request.discount,
            "profit": request.profit

        }])

        prediction = anomaly_model.predict(
            input_data
        )[0]

        status = (
            "Anomaly Detected"
            if prediction == -1
            else "Normal Transaction"
        )

        logger.info(
            f"Anomaly detection result: {status}"
        )

        # =========================================
        # SAVE TO MONGODB
        # =========================================

        anomaly_collection.insert_one({

            "sales": request.sales,
            "quantity": request.quantity,
            "discount": request.discount,
            "profit": request.profit,
            "status": status

        })

        return {

            "status": status
        }

    except Exception as e:

        logger.error(f"Anomaly API Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# CHAT AGENT API
# =========================================

@app.post("/chat")
def chat_with_agent(request: ChatRequest):

    try:

        response = coordinator_agent(
            request.query
        )

        logger.info(
            f"Chat query processed: {request.query}"
        )

        # =========================================
        # SAVE CHAT HISTORY
        # =========================================

        chat_collection.insert_one({

            "query": request.query,
            "response": response

        })

        return {

            "response": response
        }

    except Exception as e:

        logger.error(f"Chat API Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )