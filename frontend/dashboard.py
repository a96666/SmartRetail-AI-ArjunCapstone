import streamlit as st
import requests
import pandas as pd


# =====================================
# AZURE BACKEND URL
# =====================================

BASE_URL = "https://smart-retail-ai-app-eaf0ada6akevggfb.centralindia-01.azurewebsites.net"


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Smart Retail AI Platform",
    layout="wide"
)


# =====================================
# TITLE
# =====================================

st.title("Smart Retail AI Platform")

st.markdown(
    "AI Powered Retail Analytics Dashboard"
)


# =====================================
# SIDEBAR
# =====================================

menu = st.sidebar.selectbox(

    "Select Module",

    [

        "Sales Forecast",

        "Anomaly Detection",

        "AI Chat Assistant"
    ]
)


# =====================================
# SALES FORECAST
# =====================================

if menu == "Sales Forecast":

    st.header("Sales Forecast Prediction")


    region = st.selectbox(

        "Region",

        ["West", "East", "Central", "South"]
    )


    segment = st.selectbox(

        "Segment",

        ["Consumer", "Corporate", "Home Office"]
    )


    shipping_mode = st.selectbox(

        "Shipping Mode",

        [

            "First Class",

            "Second Class",

            "Standard Class"
        ]
    )


    category = st.selectbox(

        "Category",

        [

            "Technology",

            "Furniture",

            "Office Supplies"
        ]
    )


    product_name = st.text_input(

        "Product Name",

        "Laptop"
    )


    quantity = st.number_input(

        "Quantity",

        1,

        100,

        5
    )


    discount = st.slider(

        "Discount",

        0.0,

        1.0,

        0.05
    )


    if st.button("Predict Sales"):

        data = {

            "region": region,

            "segment": segment,

            "shipping_mode": shipping_mode,

            "category": category,

            "product_name": product_name,

            "quantity": quantity,

            "discount": discount,

            "year": 2025,

            "month": 5,

            "day": 22,

            "weekday": 4
        }

        try:

            response = requests.post(

                f"{BASE_URL}/predict-sales",

                json=data
            )

            result = response.json()

            st.write(result)

            if "predicted_sales" in result:

                st.success(

                    f"Predicted Sales: {result['predicted_sales']}"
                )

            elif "prediction" in result:

                st.success(

                    f"Predicted Sales: {result['prediction']}"
                )

            else:

                st.error(

                    f"Unexpected API Response: {result}"
                )

        except Exception as e:

            st.error(f"API Error: {e}")


# =====================================
# ANOMALY DETECTION
# =====================================

elif menu == "Anomaly Detection":

    st.header("Fraud / Anomaly Detection")


    sales = st.number_input(

        "Sales",

        100.0,

        500000.0,

        150000.0
    )


    quantity = st.number_input(

        "Quantity",

        1,

        50,

        2
    )


    discount = st.slider(

        "Discount",

        0.0,

        1.0,

        0.01
    )


    profit = st.number_input(

        "Profit",

        -10000.0,

        100000.0,

        100.0
    )


    if st.button("Detect Anomaly"):

        data = {

            "sales": sales,

            "quantity": quantity,

            "discount": discount,

            "profit": profit
        }

        try:

            response = requests.post(

                f"{BASE_URL}/detect-anomaly",

                json=data
            )

            result = response.json()

            st.write(result)

            if "status" in result:

                st.warning(

                    f"Status: {result['status']}"
                )

            else:

                st.error(

                    f"Unexpected API Response: {result}"
                )

        except Exception as e:

            st.error(f"API Error: {e}")


# =====================================
# AI CHAT ASSISTANT
# =====================================

else:

    st.header("AI Retail Assistant")


    query = st.text_input(

        "Ask Question"
    )


    if st.button("Ask AI"):

        data = {

            "query": query
        }

        try:

            response = requests.post(

                f"{BASE_URL}/chat",

                json=data
            )

            result = response.json()

            st.write(result)

            if "response" in result:

                st.info(result["response"])

            else:

                st.error(

                    f"Unexpected API Response: {result}"
                )

        except Exception as e:

            st.error(f"API Error: {e}")