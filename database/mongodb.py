from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"

client = MongoClient(MONGO_URL)

db = client["smart_retail_ai"]

sales_collection = db["sales_predictions"]
anomaly_collection = db["anomalies"]
chat_collection = db["chat_history"]