import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# =========================================
# LOAD ENV VARIABLES
# =========================================

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

# =========================================
# CONNECT TO MONGODB
# =========================================

client = MongoClient(MONGO_URL)

db = client["smart_retail_ai"]

sales_collection = db["sales_predictions"]

anomaly_collection = db["anomalies"]

# =========================================
# FETCH SALES DATA
# =========================================

sales_data = list(
    sales_collection.find({}, {"_id": 0})
)

sales_df = pd.DataFrame(sales_data)

# =========================================
# GENERATE SALES SUMMARY
# =========================================

if not sales_df.empty:

    region_summary = (
        sales_df.groupby("region")["predicted_sales"]
        .mean()
        .reset_index()
    )

    region_summary.to_csv(
        "data/region_sales_summary.csv",
        index=False
    )

    category_summary = (
        sales_df.groupby("category")["predicted_sales"]
        .mean()
        .reset_index()
    )

    category_summary.to_csv(
        "data/category_sales_summary.csv",
        index=False
    )

    print("Sales Summary Generated")

# =========================================
# FETCH ANOMALY DATA
# =========================================

anomaly_data = list(
    anomaly_collection.find({}, {"_id": 0})
)

anomaly_df = pd.DataFrame(anomaly_data)

# =========================================
# GENERATE ANOMALY REPORT
# =========================================

if not anomaly_df.empty:

    anomaly_summary = (
        anomaly_df["status"]
        .value_counts()
        .reset_index()
    )

    anomaly_summary.columns = [
        "status",
        "count"
    ]

    anomaly_summary.to_csv(
        "data/anomaly_summary.csv",
        index=False
    )

    print("Anomaly Summary Generated")

print("Dashboard Dataset Ready")