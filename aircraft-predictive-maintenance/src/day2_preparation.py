import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

input_file = (
    RESULTS_DIR / "FD001_processed.csv"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# Find sensors
# --------------------------------------------------

sensor_columns = [
    column
    for column in df.columns
    if column.startswith("sensor_")
]

print(
    "\nNumber of sensors:",
    len(sensor_columns)
)

# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMissing values:")

print(
    df[sensor_columns]
    .isnull()
    .sum()
)

# --------------------------------------------------
# Sensor variability
# --------------------------------------------------

sensor_std = (
    df[sensor_columns].std()
)

print("\nSensor standard deviation:")

print(
    sensor_std.sort_values()
)

# --------------------------------------------------
# Remove near-constant sensors
# --------------------------------------------------

threshold = 0.01

selected_sensors = [
    sensor
    for sensor in sensor_columns
    if sensor_std[sensor] > threshold
]

print("\nSelected sensors:")

print(selected_sensors)

print(
    "\nNumber of selected sensors:",
    len(selected_sensors)
)

# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + selected_sensors

# --------------------------------------------------
# Engine-level train/validation split
# --------------------------------------------------

engine_ids = df["unit"].unique()

np.random.seed(42)

np.random.shuffle(engine_ids)

split_index = int(
    len(engine_ids) * 0.8
)

train_engines = (
    engine_ids[:split_index]
)

validation_engines = (
    engine_ids[split_index:]
)

train_df = df[
    df["unit"].isin(train_engines)
].copy()

validation_df = df[
    df["unit"].isin(validation_engines)
].copy()

print(
    "\nTraining engines:",
    len(train_engines)
)

print(
    "Validation engines:",
    len(validation_engines)
)

print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(validation_df)
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------

scaler = MinMaxScaler()

train_df[features] = (
    scaler.fit_transform(
        train_df[features]
    )
)

validation_df[features] = (
    scaler.transform(
        validation_df[features]
    )
)

# --------------------------------------------------
# Save scaler
# --------------------------------------------------

scaler_file = (
    MODELS_DIR /
    "feature_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_file
)

print(
    "\nScaler saved:",
    scaler_file
)

# --------------------------------------------------
# Save prepared data
# --------------------------------------------------

train_output = (
    RESULTS_DIR /
    "train_prepared.csv"
)

validation_output = (
    RESULTS_DIR /
    "validation_prepared.csv"
)

train_df.to_csv(
    train_output,
    index=False
)

validation_df.to_csv(
    validation_output,
    index=False
)

print(
    "\nTraining data saved:",
    train_output
)

print(
    "Validation data saved:",
    validation_output
)