import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("Loading training data...")

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


print("\nData loaded successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print(
    "X_validation:",
    X_validation.shape
)

print(
    "y_validation:",
    y_validation.shape
)


# ==================================================
# INPUT DIMENSIONS
# ==================================================

time_steps = X_train.shape[1]

number_of_features = X_train.shape[2]

print("\nTime steps:", time_steps)

print(
    "Number of features:",
    number_of_features
)


# ==================================================
# BUILD LSTM MODEL
# ==================================================

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            time_steps,
            number_of_features
        )
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ==================================================
# DISPLAY MODEL
# ==================================================

print("\nModel architecture:")

model.summary()


# ==================================================
# EARLY STOPPING
# ==================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ==================================================
# TRAIN
# ==================================================

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=50,

    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ==================================================
# SAVE MODEL
# ==================================================

model_file = (
    MODELS_DIR /
    "aircraft_rul_lstm.keras"
)

model.save(model_file)

print(
    "\nModel saved to:"
)

print(model_file)


# ==================================================
# PLOT LOSS
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "LSTM Training and Validation Loss"
)

plt.legend()

plt.grid(True)

loss_file = (
    RESULTS_DIR /
    "lstm_training_loss.png"
)

plt.savefig(
    loss_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nLoss graph saved to:"
)

print(loss_file)