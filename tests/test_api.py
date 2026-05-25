from fastapi.testclient import TestClient

from app.api.main_api import app

client = TestClient(app)


# =========================================
# HOME API TEST
# =========================================

def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Smart Retail AI Platform Running"
    }


# =========================================
# SALES PREDICTION TEST
# =========================================

def test_predict_sales():

    payload = {

        "region": "North",
        "segment": "Consumer",
        "shipping_mode": "Standard",
        "category": "Electronics",
        "product_name": "Laptop",
        "quantity": 5,
        "discount": 10,
        "year": 2025,
        "month": 5,
        "day": 25,
        "weekday": 1
    }

    response = client.post(
        "/predict-sales",
        json=payload
    )

    assert response.status_code == 200

    assert "predicted_sales" in response.json()


# =========================================
# ANOMALY TEST
# =========================================

def test_detect_anomaly():

    payload = {

        "sales": 4500,
        "quantity": 5,
        "discount": 10,
        "profit": 500
    }

    response = client.post(
        "/detect-anomaly",
        json=payload
    )

    assert response.status_code == 200

    assert "status" in response.json()


# =========================================
# CHAT API TEST
# =========================================

def test_chat():

    payload = {

        "query": "What are the top products?"
    }

    response = client.post(
        "/chat",
        json=payload
    )

    assert response.status_code == 200

    assert "response" in response.json()