import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

train_file = (
    RESULTS_DIR / "train_prepared.csv"
)

validation_file = (
    RESULTS_DIR / "validation_prepared.csv"
)

train_df = pd.read_csv(train_file)

validation_df = pd.read_csv(
    validation_file
)

# Features
features = [
    column
    for column in train_df.columns
    if (
        column.startswith("sensor_")
        or column.startswith("setting_")
        or column == "cycle"
    )
]

# Number of previous cycles
WINDOW_SIZE = 30


def create_sequences(
    data,
    features,
    window_size
):

    X = []
    y = []

    for engine_id in data["unit"].unique():

        engine = data[
            data["unit"] == engine_id
        ].sort_values("cycle")

        feature_values = (
            engine[features].values
        )

        rul_values = (
            engine["RUL"].values
        )

        for i in range(
            window_size,
            len(engine)
        ):

            sequence = feature_values[
                i - window_size:i
            ]

            target = rul_values[i]

            X.append(sequence)

            y.append(target)

    return (
        np.array(X),
        np.array(y)
    )


# Training sequences
X_train, y_train = create_sequences(
    train_df,
    features,
    WINDOW_SIZE
)

# Validation sequences
X_validation, y_validation = (
    create_sequences(
        validation_df,
        features,
        WINDOW_SIZE
    )
)

print("\nX_train shape:")
print(X_train.shape)

print("\ny_train shape:")
print(y_train.shape)

print("\nX_validation shape:")
print(X_validation.shape)

print("\ny_validation shape:")
print(y_validation.shape)

# Save
np.save(
    RESULTS_DIR / "X_train.npy",
    X_train
)

np.save(
    RESULTS_DIR / "y_train.npy",
    y_train
)

np.save(
    RESULTS_DIR / "X_validation.npy",
    X_validation
)

np.save(
    RESULTS_DIR / "y_validation.npy",
    y_validation
)

print(
    "\nLSTM sequences saved successfully."
)