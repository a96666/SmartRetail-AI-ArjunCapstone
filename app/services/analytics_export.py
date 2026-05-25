import pandas as pd


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    "data/enterprise_retail_dataset.csv"
)


print("\nDataset Loaded Successfully!")


# =========================================
# SALES BY REGION
# =========================================

region_sales = (

    df.groupby("region")["sales"]

    .sum()

    .reset_index()

    .sort_values(
        by="sales",
        ascending=False
    )
)


region_sales.to_csv(

    "data/region_sales_summary.csv",

    index=False
)


print("\nRegion sales summary exported!")


# =========================================
# SALES BY CATEGORY
# =========================================

category_sales = (

    df.groupby("category")["sales"]

    .sum()

    .reset_index()

    .sort_values(
        by="sales",
        ascending=False
    )
)


category_sales.to_csv(

    "data/category_sales_summary.csv",

    index=False
)


print("\nCategory sales summary exported!")


# =========================================
# PROFIT BY REGION
# =========================================

profit_summary = (

    df.groupby("region")["profit"]

    .sum()

    .reset_index()

    .sort_values(
        by="profit",
        ascending=False
    )
)


profit_summary.to_csv(

    "data/profit_summary.csv",

    index=False
)


print("\nProfit summary exported!")


# =========================================
# MONTHLY SALES TREND
# =========================================

# Create month column if missing

if "month" not in df.columns:

    df["month"] = pd.to_datetime(
        df["order_date"]
    ).dt.month


monthly_sales = (

    df.groupby("month")["sales"]

    .sum()

    .reset_index()

    .sort_values(
        by="month"
    )
)


monthly_sales.to_csv(

    "data/monthly_sales_trend.csv",

    index=False
)


print("\nMonthly sales trend exported!")

# =========================================
# ANOMALY REPORT
# =========================================

anomaly_report = (

    df["anomaly"]

    .value_counts()

    .reset_index()
)

anomaly_report.columns = [

    "anomaly_status",

    "count"
]


anomaly_report.to_csv(

    "data/anomaly_summary.csv",

    index=False
)


print("\nAnomaly summary exported!")


# =========================================
# TOP PRODUCTS
# =========================================

top_products = (

    df.groupby("product_name")["sales"]

    .sum()

    .reset_index()

    .sort_values(
        by="sales",
        ascending=False
    )

    .head(10)
)


top_products.to_csv(

    "data/top_products.csv",

    index=False
)


print("\nTop products report exported!")


# =========================================
# FINAL MESSAGE
# =========================================

print("\nAll analytics reports exported successfully!")