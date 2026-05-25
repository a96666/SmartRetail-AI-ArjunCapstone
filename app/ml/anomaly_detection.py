import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    "data/enterprise_retail_dataset.csv"
)


# =========================================
# FEATURE SELECTION
# =========================================

features = [

    "sales",

    "quantity",

    "discount",

    "profit"
]


X = df[features]


# =========================================
# CREATE ANOMALY PIPELINE
# =========================================

anomaly_pipeline = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",

        IsolationForest(

            n_estimators=300,

            contamination=0.015,

            random_state=42
        )
    )
])


# =========================================
# TRAIN MODEL
# =========================================

print("\nTraining anomaly detection model...")

anomaly_pipeline.fit(X)

print("\nAnomaly model training completed!")


# =========================================
# PREDICT ANOMALIES
# =========================================

predictions = anomaly_pipeline.predict(X)


# =========================================
# CONVERT PREDICTIONS
# =========================================

# Isolation Forest:
# -1 = anomaly
#  1 = normal

df["anomaly_prediction"] = predictions


df["anomaly_prediction"] = df[
    "anomaly_prediction"
].map({

    1: 0,

    -1: 1
})


# =========================================
# ANOMALY SCORE
# =========================================

scores = anomaly_pipeline.named_steps[
    "model"
].decision_function(

    anomaly_pipeline.named_steps[
        "scaler"
    ].transform(X)
)


df["anomaly_score"] = scores


# =========================================
# EXTRACT ANOMALIES
# =========================================

anomalies_df = df[
    df["anomaly_prediction"] == 1
]


# =========================================
# DISPLAY RESULTS
# =========================================

print("\nTotal Records:")
print(len(df))

print("\nTotal Anomalies Detected:")
print(len(anomalies_df))


print("\nFirst 5 Anomalies:")
print(anomalies_df.head())


# =========================================
# SAVE REPORTS
# =========================================

df.to_csv(

    "data/final_anomaly_report.csv",

    index=False
)


anomalies_df.to_csv(

    "data/detected_anomalies.csv",

    index=False
)


# =========================================
# SAVE MODEL
# =========================================

joblib.dump(

    anomaly_pipeline,

    "models/anomaly_detection_model.pkl"
)


# =========================================
# FINAL MESSAGE
# =========================================

print("\nFinal anomaly report saved successfully!")

print("\nDetected anomalies dataset saved successfully!")

print("\nAnomaly detection model saved successfully!")