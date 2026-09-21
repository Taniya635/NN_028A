import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# Load training data
# ============================================================

train_df = pd.read_csv(
    RESULTS_DIR / "FD001_processed.csv"
)


# ============================================================
# Select the same sensors used during training
# ============================================================

sensor_columns = [
    column
    for column in train_df.columns
    if column.startswith("sensor_")
]

sensor_std = train_df[sensor_columns].std()

selected_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor_std[sensor] > 0.01
]

features = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + selected_sensors


# ============================================================
# Fit scaler using training data
# ============================================================

scaler = MinMaxScaler()

scaler.fit(
    train_df[features]
)


# ============================================================
# Load trained LSTM
# ============================================================

model = load_model(
    MODELS_DIR / "aircraft_rul_lstm.keras"
)


# ============================================================
# Load test engine data
# ============================================================

test_file = BASE_DIR / "data" / "test_FD001.txt"

columns = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_df = pd.read_csv(
    test_file,
    sep=r"\s+",
    header=None,
    names=columns
)


# ============================================================
# Select an engine
# ============================================================

engine_id = 1

engine = test_df[
    test_df["unit"] == engine_id
].sort_values("cycle").copy()


# ============================================================
# Check sufficient history
# ============================================================

WINDOW_SIZE = 30

if len(engine) < WINDOW_SIZE:
    raise ValueError(
        f"Engine {engine_id} has only {len(engine)} cycles. "
        f"At least {WINDOW_SIZE} cycles are required."
    )


# ============================================================
# Get latest 30 cycles
# ============================================================

latest_data = engine[features].tail(
    WINDOW_SIZE
)


# ============================================================
# Scale
# ============================================================

scaled_data = scaler.transform(
    latest_data.to_numpy()
)


# ============================================================
# Create LSTM input
# ============================================================

X = scaled_data.reshape(
    1,
    WINDOW_SIZE,
    len(features)
)


# ============================================================
# Predict
# ============================================================

prediction = model.predict(
    X,
    verbose=0
)[0][0]


# ============================================================
# Prevent negative RUL
# ============================================================

prediction = max(
    0,
    prediction
)


# ============================================================
# Display result
# ============================================================

current_cycle = int(
    engine["cycle"].iloc[-1]
)

print("\n======================================")
print("AIRCRAFT RUL PREDICTION")
print("======================================")

print(f"Engine ID       : {engine_id}")
print(f"Current Cycle   : {current_cycle}")
print(f"Predicted RUL   : {prediction:.2f} cycles")

print("======================================")