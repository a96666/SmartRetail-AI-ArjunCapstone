from database.mongodb import sales_collection

def save_prediction(data):
    sales_collection.insert_one(data)