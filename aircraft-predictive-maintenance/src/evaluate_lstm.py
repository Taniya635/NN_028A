import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR = BASE_DIR / "models"


# ==================================================
# LOAD DATA
# ==================================================

X_validation = np.load(
    RESULTS_DIR / "X_validation.npy"
)

y_validation = np.load(
    RESULTS_DIR / "y_validation.npy"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = load_model(
    MODELS_DIR /
    "aircraft_rul_lstm.keras"
)


print("Model loaded successfully.")


# ==================================================
# PREDICTION
# ==================================================

predicted_rul = model.predict(
    X_validation,
    verbose=1
).flatten()


# ==================================================
# METRICS
# ==================================================

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


print("\n==============================")
print("       MODEL PERFORMANCE")
print("==============================")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)


# ==================================================
# ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(12, 6))

sample_size = min(
    500,
    len(y_validation)
)

plt.plot(
    y_validation[:sample_size],
    label="Actual RUL"
)

plt.plot(
    predicted_rul[:sample_size],
    label="Predicted RUL"
)

plt.xlabel(
    "Validation Sample"
)

plt.ylabel(
    "Remaining Useful Life"
)

plt.title(
    "Actual vs Predicted Aircraft Engine RUL"
)

plt.legend()

plt.grid(True)


output_file = (
    RESULTS_DIR /
    "actual_vs_predicted_rul.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nGraph saved to:"
)

print(output_file)