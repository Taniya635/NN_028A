import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

X_train = np.load(
    RESULTS_DIR / "X_train.npy"
)

y_train = np.load(
    RESULTS_DIR / "y_train.npy"
)

X_validation = np.load(
    RESULTS_DIR / "X_validation.npy"
)

y_validation = np.load(
    RESULTS_DIR / "y_validation.npy"
)

print("================================")
print("       DATASET CHECK")
print("================================")

print(
    "\nX_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_validation:",
    X_validation.shape
)

print(
    "y_validation:",
    y_validation.shape
)

print(
    "\nFirst training RUL:",
    y_train[0]
)

print(
    "Minimum training RUL:",
    y_train.min()
)

print(
    "Maximum training RUL:",
    y_train.max()
)

print(
    "\nNaN in X_train:",
    np.isnan(X_train).sum()
)

print(
    "NaN in y_train:",
    np.isnan(y_train).sum()
)

print(
    "\nNaN in X_validation:",
    np.isnan(X_validation).sum()
)

print(
    "NaN in y_validation:",
    np.isnan(y_validation).sum()
)

print(
    "\n================================"
)

print(
    "Data preparation complete!"
)

print(
    "================================"
)