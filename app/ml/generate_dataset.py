import pandas as pd
import numpy as np
import random
from faker import Faker


# =========================================
# INITIAL SETUP
# =========================================

fake = Faker()

np.random.seed(42)
random.seed(42)


# =========================================
# CONFIGURATION
# =========================================

NUM_ROWS = 15000


# =========================================
# CATEGORY PRODUCT MAPPING
# =========================================

categories = {

    "Electronics": {

        "products": [
            "Laptop",
            "Smartphone",
            "Tablet",
            "Monitor"
        ],

        "base_price": 5000
    },

    "Furniture": {

        "products": [
            "Chair",
            "Desk",
            "Sofa",
            "Table"
        ],

        "base_price": 2500
    },

    "Office Supplies": {

        "products": [
            "Notebook",
            "Printer",
            "Pen",
            "Storage Box"
        ],

        "base_price": 500
    }
}


# =========================================
# REGIONS
# =========================================

regions = {

    "North": 1.10,

    "South": 0.95,

    "East": 0.90,

    "West": 1.20
}


# =========================================
# CUSTOMER SEGMENTS
# =========================================

segments = {

    "Consumer": 1.0,

    "Corporate": 1.3,

    "Home Office": 0.85
}


# =========================================
# SHIPPING MODES
# =========================================

shipping_modes = {

    "Standard Class": 1.0,

    "Second Class": 1.05,

    "First Class": 1.12,

    "Same Day": 1.20
}


# =========================================
# DATA STORAGE
# =========================================

data = []


# =========================================
# GENERATE DATA
# =========================================

for _ in range(NUM_ROWS):

    # Category selection
    category = random.choice(
        list(categories.keys())
    )

    product = random.choice(
        categories[category]["products"]
    )

    base_price = categories[category][
        "base_price"
    ]


    # Date
    order_date = fake.date_between(
        start_date="-3y",
        end_date="today"
    )

    month = order_date.month


    # Region & segment
    region = random.choice(
        list(regions.keys())
    )

    segment = random.choice(
        list(segments.keys())
    )

    shipping_mode = random.choice(
        list(shipping_modes.keys())
    )


    # Quantity
    quantity = random.randint(1, 8)


    # Discount
    discount = round(
        random.uniform(0, 0.25),
        2
    )


    # =====================================
    # SALES FORMULA
    # =====================================

    sales = (

        base_price

        * quantity

        * regions[region]

        * segments[segment]

        * shipping_modes[shipping_mode]

        * (1 - discount)
    )


    # =====================================
    # SEASONAL EFFECT
    # =====================================

    if month in [10, 11, 12]:

        sales *= 1.30


    elif month in [6, 7]:

        sales *= 0.90


    # =====================================
    # CONTROLLED NOISE
    # =====================================

    sales += np.random.normal(0, 300)


    # Avoid negative values
    sales = max(100, sales)


    # =====================================
    # PROFIT
    # =====================================

    profit_margin = random.uniform(
        0.12,
        0.30
    )

    profit = sales * profit_margin


    # =====================================
    # ANOMALIES
    # =====================================

    anomaly = 0

    if random.random() < 0.015:

        sales *= random.uniform(2.5, 4)

        anomaly = 1


    # =====================================
    # FINAL ROW
    # =====================================

    data.append({

        "order_date": order_date,

        "region": region,

        "segment": segment,

        "shipping_mode": shipping_mode,

        "category": category,

        "product_name": product,

        "quantity": quantity,

        "discount": discount,

        "sales": round(sales, 2),

        "profit": round(profit, 2),

        "anomaly": anomaly
    })


# =========================================
# DATAFRAME
# =========================================

df = pd.DataFrame(data)


# =========================================
# SAVE DATASET
# =========================================

df.to_csv(

    "data/enterprise_retail_dataset.csv",

    index=False
)


# =========================================
# DISPLAY RESULTS
# =========================================

print("\nEnterprise dataset generated successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nAnomaly Distribution:")
print(df["anomaly"].value_counts())