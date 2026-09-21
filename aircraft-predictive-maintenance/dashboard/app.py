import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# Title
# ============================================================

st.title("✈️ AI-Powered Aircraft Predictive Maintenance")

st.markdown(
    """
    ### Remaining Useful Life (RUL) Prediction

    This system uses an **LSTM deep-learning model** trained on
    the NASA C-MAPSS FD001 dataset to estimate the Remaining
    Useful Life of an aircraft engine.
    """
)


# ============================================================
# Load training data
# ============================================================

@st.cache_data
def load_training_data():

    return pd.read_csv(
        RESULTS_DIR / "FD001_processed.csv"
    )


# ============================================================
# Load test data
# ============================================================

@st.cache_data
def load_test_data():

    columns = (
        ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    return pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=columns
    )


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_lstm_model():

    return load_model(
        MODELS_DIR / "aircraft_rul_lstm.keras"
    )


train_df = load_training_data()
test_df = load_test_data()
model = load_lstm_model()


# ============================================================
# Feature selection
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
# Scaler
# ============================================================

scaler = MinMaxScaler()

scaler.fit(
    train_df[features]
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Engine Selection")

engine_ids = sorted(
    test_df["unit"].unique()
)

engine_id = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)


# ============================================================
# Selected engine
# ============================================================

engine = test_df[
    test_df["unit"] == engine_id
].sort_values("cycle")


WINDOW_SIZE = 30


if len(engine) < WINDOW_SIZE:

    st.error(
        "This engine does not have enough cycles "
        "for prediction."
    )

    st.stop()


# ============================================================
# Latest sequence
# ============================================================

latest = engine[
    features
].tail(WINDOW_SIZE)


scaled = scaler.transform(
    latest.to_numpy()
)


X = scaled.reshape(
    1,
    WINDOW_SIZE,
    len(features)
)


# ============================================================
# Prediction
# ============================================================

prediction = model.predict(
    X,
    verbose=0
)[0][0]

prediction = max(
    0,
    float(prediction)
)


current_cycle = int(
    engine["cycle"].iloc[-1]
)


# ============================================================
# Maintenance status
# ============================================================

if prediction <= 20:

    status = "🔴 Immediate Maintenance Attention"

elif prediction <= 50:

    status = "🟠 Maintenance Monitoring"

else:

    status = "🟢 Normal Monitoring"


# ============================================================
# Main metrics
# ============================================================

st.subheader("Engine Health Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Engine ID",
        engine_id
    )

with col2:

    st.metric(
        "Current Cycle",
        current_cycle
    )

with col3:

    st.metric(
        "Predicted RUL",
        f"{prediction:.2f} cycles"
    )


st.info(
    f"Maintenance Status: {status}"
)


# ============================================================
# Sensor trends
# ============================================================

st.subheader("Recent Sensor Measurements")

sensor_display = [
    sensor
    for sensor in selected_sensors[:6]
]

sensor_data = engine[
    ["cycle"] + sensor_display
].tail(50)

st.line_chart(
    sensor_data.set_index("cycle")
)


# ============================================================
# Model information
# ============================================================

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.write("**Model:** LSTM")

    st.write(
        "**Dataset:** NASA C-MAPSS FD001"
    )

    st.write(
        "**Sequence Length:** 30 cycles"
    )

with col2:

    st.write(
        "**Task:** Remaining Useful Life Prediction"
    )

    st.write(
        "**Input:** Multivariate engine sensor data"
    )

    st.write(
        "**Output:** Predicted RUL in cycles"
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AI-Powered Predictive Maintenance for Aircraft | "
    "BCA Final Year Project"
)