import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load prepared data
# --------------------------------------------------

train_df = pd.read_csv(RESULTS_DIR / "train_prepared.csv")
validation_df = pd.read_csv(RESULTS_DIR / "validation_prepared.csv")


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

features = [
    column
    for column in train_df.columns
    if (
        column.startswith("sensor_")
        or column.startswith("setting_")
        or column == "cycle"
    )
]

WINDOW_SIZE = 30


# --------------------------------------------------
# Convert time-series windows into statistical features
# --------------------------------------------------

def create_xgboost_features(data, features, window_size):

    X = []
    y = []

    for engine_id in data["unit"].unique():

        engine = data[data["unit"] == engine_id].sort_values("cycle")

        values = engine[features].values
        rul_values = engine["RUL"].values

        for i in range(window_size, len(engine)):

            window = values[i - window_size:i]

            feature_vector = []

            # Mean
            feature_vector.extend(np.mean(window, axis=0))

            # Standard deviation
            feature_vector.extend(np.std(window, axis=0))

            # Minimum
            feature_vector.extend(np.min(window, axis=0))

            # Maximum
            feature_vector.extend(np.max(window, axis=0))

            # Last value in the window
            feature_vector.extend(window[-1])

            X.append(feature_vector)
            y.append(rul_values[i])

    return np.array(X), np.array(y)


print("Creating XGBoost training features...")

X_train, y_train = create_xgboost_features(
    train_df,
    features,
    WINDOW_SIZE
)

print("Creating XGBoost validation features...")

X_validation, y_validation = create_xgboost_features(
    validation_df,
    features,
    WINDOW_SIZE
)


print("\nTraining data shape:")
print(X_train.shape)

print("Validation data shape:")
print(X_validation.shape)


# --------------------------------------------------
# XGBoost model
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    verbose=False
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

predicted_rul = model.predict(X_validation)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_validation,
    predicted_rul
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        predicted_rul
    )
)

print("\nXGBoost Results")
print("------------------------")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_file = MODELS_DIR / "xgboost_rul_model.pkl"

joblib.dump(
    model,
    model_file
)

print(f"\nModel saved to:")
print(model_file)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_RUL": y_validation,
    "Predicted_RUL": predicted_rul
})

prediction_file = RESULTS_DIR / "xgboost_predictions.csv"

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(f"Predictions saved to:")
print(prediction_file)