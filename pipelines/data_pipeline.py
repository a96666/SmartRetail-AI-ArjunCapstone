import pandas as pd
import os


# =========================================
# CREATE FOLDERS
# =========================================

os.makedirs("data/staged", exist_ok=True)
os.makedirs("data/curated", exist_ok=True)


# =========================================
# LOAD RAW DATA
# =========================================

raw_data = pd.read_csv(
    "data/raw/enterprise_retail_dataset.csv"
)

print("Raw Data Loaded")


# =========================================
# DATA CLEANING
# =========================================

staged_data = raw_data.copy()

# Remove duplicates
staged_data.drop_duplicates(
    inplace=True
)

# Fill missing values
staged_data.fillna(
    0,
    inplace=True
)

print("Data Cleaning Completed")


# =========================================
# FEATURE ENGINEERING
# =========================================

staged_data["sales_per_quantity"] = (

    staged_data["sales"]
    /
    staged_data["quantity"]
)

print("Feature Engineering Completed")


# =========================================
# SAVE STAGED DATA
# =========================================

staged_data.to_csv(

    "data/staged/staged_retail_data.csv",

    index=False
)

print("Staged Data Saved")


# =========================================
# CURATED DATA
# =========================================

curated_data = staged_data.copy()

# Keep only positive sales
curated_data = curated_data[
    curated_data["sales"] > 0
]

print("Curated Dataset Prepared")


# =========================================
# SAVE CURATED DATA
# =========================================

curated_data.to_parquet(

    "data/curated/curated_retail_data.parquet",

    index=False
)

print("Curated Data Saved")


print("Data Pipeline Completed Successfully")