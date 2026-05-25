import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    "data/enterprise_retail_dataset.csv"
)


# =========================================
# FEATURE ENGINEERING
# =========================================

df["order_date"] = pd.to_datetime(
    df["order_date"]
)

df["year"] = df["order_date"].dt.year

df["month"] = df["order_date"].dt.month

df["day"] = df["order_date"].dt.day

df["weekday"] = df["order_date"].dt.weekday


# =========================================
# FEATURE SELECTION
# =========================================

features = [

    "region",

    "segment",

    "shipping_mode",

    "category",

    "product_name",

    "quantity",

    "discount",

    "year",

    "month",

    "day",

    "weekday"
]


target = "sales"


X = df[features]

y = df[target]


# =========================================
# CATEGORICAL FEATURES
# =========================================

categorical_features = [

    "region",

    "segment",

    "shipping_mode",

    "category",

    "product_name"
]


# =========================================
# PREPROCESSING
# =========================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "categorical",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features
        )
    ],

    remainder="passthrough"
)


# =========================================
# TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)


# =========================================
# MODEL COLLECTION
# =========================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}


# =========================================
# STORE RESULTS
# =========================================

results = []

best_model = None

best_score = -999

best_pipeline = None


# =========================================
# TRAIN & EVALUATE MODELS
# =========================================

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")


    pipeline = Pipeline([

        ("preprocessing", preprocessor),

        ("model", model)
    ])


    # Train
    pipeline.fit(X_train, y_train)


    # Predict
    predictions = pipeline.predict(X_test)


    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    # Save results
    results.append({

        "Model": model_name,

        "MAE": round(mae, 2),

        "R2 Score": round(r2, 4)
    })


    print(f"{model_name} MAE: {mae}")

    print(f"{model_name} R2 Score: {r2}")


    # Select best model
    if r2 > best_score:

        best_score = r2

        best_model = model_name

        best_pipeline = pipeline


# =========================================
# RESULTS DATAFRAME
# =========================================

results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")

print(results_df)


# =========================================
# BEST MODEL
# =========================================

print(f"\nBest Model: {best_model}")

print(f"Best R2 Score: {best_score}")


# =========================================
# SAVE BEST MODEL
# =========================================

joblib.dump(

    best_pipeline,

    "models/best_forecasting_model.pkl"
)


# =========================================
# SAVE RESULTS
# =========================================

results_df.to_csv(

    "data/model_comparison_results.csv",

    index=False
)


print("\nBest model saved successfully!")

print("\nModel comparison report saved successfully!")