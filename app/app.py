# ==============================
# app.py - Streamlit App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import tempfile
import os

st.set_page_config(page_title="Predicción Consumo Energía", layout="wide")
st.title("📊 Predicción de Consumo de Energía")

# -------------------------------------
# Función para cargar el modelo desde GitHub
# -------------------------------------
@st.cache_data(show_spinner=True)
def cargar_modelo():
    url = "https://raw.githubusercontent.com/Maheferrer98/mi_app_streamlit/main/app/modelo_xgb_500k.pkl"
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        modelo = joblib.load(tmp_path)
        os.remove(tmp_path)
        return modelo
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        return None

model = cargar_modelo()
if model:
    st.success("Modelo cargado correctamente ✅")
else:
    st.stop()

# -------------------------------------
# Inputs del usuario
# -------------------------------------
st.header("Ingrese los valores para predecir el consumo")

def input_features():
    st.subheader("Variables numéricas estandarizadas")
    global_reactive_power = st.number_input("Global Reactive Power", min_value=0.0, step=0.01, value=0.1)
    voltage = st.number_input("Voltage", min_value=0.0, step=0.1, value=235.0)
    global_intensity = st.number_input("Global Intensity", min_value=0.0, step=0.1, value=1.0)
    sub_metering_1 = st.number_input("Sub Metering 1 (Cocina)", min_value=0.0, step=0.1, value=0.0)
    sub_metering_2 = st.number_input("Sub Metering 2 (Lavandería)", min_value=0.0, step=0.1, value=0.0)
    sub_metering_3 = st.number_input("Sub Metering 3 (Agua Caliente/AC)", min_value=0.0, step=0.1, value=0.0)

    st.subheader("Variables temporales")
    hour = st.slider("Hora del día", 0, 23, 12)
    day_of_week = st.slider("Día de la semana (Lunes=0, Domingo=6)", 0, 6, 2)
    month = st.slider("Mes", 1, 12, 6)
    is_weekend = 1 if day_of_week >= 5 else 0

    st.subheader("Features derivadas")
    GAP_rolling_mean_60 = st.number_input("Media móvil 60 pasos", min_value=0.0, step=0.01, value=0.5)
    GAP_rolling_mean_120 = st.number_input("Media móvil 120 pasos", min_value=0.0, step=0.01, value=0.5)
    GAP_diff_1 = st.number_input("Diferencia 1 paso", min_value=-10.0, max_value=10.0, step=0.01, value=0.0)
    GAP_diff_60 = st.number_input("Diferencia 60 pasos", min_value=-10.0, max_value=10.0, step=0.01, value=0.0)
    sub_metering_total = sub_metering_1 + sub_metering_2 + sub_metering_3

    features = pd.DataFrame({
        "Voltage": [voltage],
        "Global_intensity": [global_intensity],
        "Sub_metering_1": [sub_metering_1],
        "Sub_metering_2": [sub_metering_2],
        "Sub_metering_3": [sub_metering_3],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
    })
    return features

input_df = input_features()

# -------------------------------------
# Predicción
# -------------------------------------
if st.button("Predecir Consumo Global Activo"):
    pred = model.predict(input_df)[0]
    st.success(f"🔹 Predicción de Consumo Global Activo: {pred:.4f} kW")
    st.info("Esta predicción está basada en el modelo XGBoost entrenado con 500k registros del dataset de consumo energético.")
