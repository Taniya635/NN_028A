import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load LSTM validation data
# --------------------------------------------------

X_validation = np.load(
    RESULTS_DIR / "X_validation.npy"
)

y_validation = np.load(
    RESULTS_DIR / "y_validation.npy"
)


# --------------------------------------------------
# Load LSTM
# --------------------------------------------------

print("Loading LSTM model...")

lstm_model = load_model(
    MODELS_DIR / "aircraft_rul_lstm.keras"
)

lstm_predictions = lstm_model.predict(
    X_validation,
    verbose=0
).flatten()


# --------------------------------------------------
# LSTM metrics
# --------------------------------------------------

lstm_mae = mean_absolute_error(
    y_validation,
    lstm_predictions
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        lstm_predictions
    )
)


# --------------------------------------------------
# Load XGBoost predictions
# --------------------------------------------------

print("Loading XGBoost predictions...")

xgb_df = pd.read_csv(
    RESULTS_DIR / "xgboost_predictions.csv"
)

xgb_actual = xgb_df["Actual_RUL"].values
xgb_predictions = xgb_df["Predicted_RUL"].values


# --------------------------------------------------
# XGBoost metrics
# --------------------------------------------------

xgb_mae = mean_absolute_error(
    xgb_actual,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        xgb_actual,
        xgb_predictions
    )
)


# --------------------------------------------------
# Comparison table
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "LSTM",
        "XGBoost"
    ],
    "MAE": [
        lstm_mae,
        xgb_mae
    ],
    "RMSE": [
        lstm_rmse,
        xgb_rmse
    ]
})


print("\nModel Comparison")
print("============================")
print(comparison.to_string(index=False))


# --------------------------------------------------
# Save comparison
# --------------------------------------------------

comparison_file = RESULTS_DIR / "model_comparison.csv"

comparison.to_csv(
    comparison_file,
    index=False
)

print(f"\nComparison saved to:")
print(comparison_file)


# --------------------------------------------------
# Create comparison chart
# --------------------------------------------------

x = np.arange(len(comparison))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    comparison["MAE"],
    width,
    label="MAE"
)

plt.bar(
    x + width / 2,
    comparison["RMSE"],
    width,
    label="RMSE"
)

plt.xticks(
    x,
    comparison["Model"]
)

plt.ylabel("Error")
plt.title("LSTM vs XGBoost Performance")

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

chart_file = RESULTS_DIR / "model_comparison.png"

plt.savefig(
    chart_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Comparison chart saved to:")
print(chart_file)
