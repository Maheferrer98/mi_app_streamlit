# ==============================
# app.py - Streamlit App (versión simplificada)
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
    st.subheader("Variables principales")
    voltage = st.number_input("Voltage", min_value=0.0, step=0.1, value=235.0)
    global_intensity = st.number_input("Global Intensity (A)", min_value=0.0, step=0.1, value=1.0)
    sub_metering_1 = st.number_input("Consumo de la Cocina", min_value=0.0, step=0.1, value=0.0)
    sub_metering_2 = st.number_input("Consumo de la Lavandería", min_value=0.0, step=0.1, value=0.0)
    sub_metering_3 = st.number_input("Consumo del Agua Caliente y Aire Acondicionado)", min_value=0.0, step=0.1, value=0.0)

    st.subheader("Variables temporales")
    hour = st.slider("Hora del día", 0, 23, 12)
    day_of_week = st.slider("Día de la semana (Lunes=0, Domingo=6)", 0, 6, 2)
    month = st.slider("Mes", 1, 12, 6)
    is_weekend = 1 if day_of_week >= 5 else 0

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
        "sub_metering_total": [sub_metering_total]
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

