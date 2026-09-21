import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# Dataset columns
# ============================================================

columns = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# Load TRAINING data
# ============================================================

train_df = pd.read_csv(
    RESULTS_DIR / "FD001_processed.csv"
)


# ============================================================
# Select sensors exactly as Day 2
# ============================================================

sensor_columns = [
    column
    for column in train_df.columns
    if column.startswith("sensor_")
]

sensor_std = train_df[sensor_columns].std()

threshold = 0.01

selected_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor_std[sensor] > threshold
]


features = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + selected_sensors


print("Selected features:")
print(features)

print("\nNumber of features:", len(features))


# ============================================================
# Recreate scaler using TRAINING data
# ============================================================

scaler = MinMaxScaler()

scaler.fit(
    train_df[features]
)

print("\nScaler recreated from training data.")


# ============================================================
# Load official TEST data
# ============================================================

test_df = pd.read_csv(
    DATA_DIR / "test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)


# ============================================================
# Load official RUL values
# ============================================================

true_rul = pd.read_csv(
    DATA_DIR / "RUL_FD001.txt",
    header=None,
    names=["RUL"]
)


print("\nTest engines:", test_df["unit"].nunique())
print("Official RUL values:", len(true_rul))


# ============================================================
# Scale test data
# ============================================================

test_df[features] = scaler.transform(
    test_df[features].to_numpy()
)


# ============================================================
# Create final 30-cycle sequence per engine
# ============================================================

WINDOW_SIZE = 30

X_test = []
engine_ids = []

for engine_id in sorted(test_df["unit"].unique()):

    engine = test_df[
        test_df["unit"] == engine_id
    ].sort_values("cycle")

    values = engine[features].to_numpy()

    if len(values) < WINDOW_SIZE:
        print(
            f"Skipping engine {engine_id}: "
            f"only {len(values)} cycles"
        )
        continue

    sequence = values[-WINDOW_SIZE:]

    X_test.append(sequence)
    engine_ids.append(engine_id)


X_test = np.asarray(X_test)


print("\nTest sequence shape:")
print(X_test.shape)


# ============================================================
# Verify shape
# ============================================================

if X_test.shape[0] != len(true_rul):

    raise ValueError(
        f"Mismatch: {X_test.shape[0]} test sequences "
        f"but {len(true_rul)} RUL values."
    )


# ============================================================
# Load LSTM
# ============================================================

print("\nLoading LSTM model...")

model = load_model(
    MODELS_DIR / "aircraft_rul_lstm.keras"
)


# ============================================================
# Prediction
# ============================================================

print("Predicting RUL...")

predicted_rul = model.predict(
    X_test,
    verbose=1
).flatten()


# ============================================================
# Results
# ============================================================

results = pd.DataFrame({
    "unit": engine_ids,
    "Actual_RUL": true_rul["RUL"].to_numpy(),
    "Predicted_RUL": predicted_rul
})


# ============================================================
# Metrics
# ============================================================

mae = mean_absolute_error(
    results["Actual_RUL"],
    results["Predicted_RUL"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["Actual_RUL"],
        results["Predicted_RUL"]
    )
)


# ============================================================
# Print results
# ============================================================

print("\n========================================")
print("NASA C-MAPSS FD001 OFFICIAL TEST")
print("========================================")

print("Test engines:", len(results))

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# Save results
# ============================================================

output_file = (
    RESULTS_DIR /
    "official_test_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)