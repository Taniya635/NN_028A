import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Dataset
train_file = DATA_DIR / "train_FD001.txt"

# Column names
columns = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load data
df = pd.read_csv(
    train_file,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

# Number of engines
number_of_engines = df["unit"].nunique()

print("\nNumber of engines:", number_of_engines)

# Maximum cycle for each engine
max_cycles = (
    df.groupby("unit")["cycle"]
    .max()
    .reset_index()
)

max_cycles.rename(
    columns={"cycle": "max_cycle"},
    inplace=True
)

# Add maximum cycle
df = df.merge(
    max_cycles,
    on="unit",
    how="left"
)

# Calculate RUL
df["RUL"] = (
    df["max_cycle"] - df["cycle"]
)

# Remove temporary column
df.drop(
    columns=["max_cycle"],
    inplace=True
)

print("\nRUL example:")
print(
    df[["unit", "cycle", "RUL"]].head(20)
)

# Save
output_file = (
    RESULTS_DIR / "FD001_processed.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nProcessed dataset saved:")
print(output_file)

print("\nFinal shape:", df.shape)