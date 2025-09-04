# ==============================
# APP STREAMLIT PARA CONSUMO ELÉCTRICO
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# 1️⃣ Título y descripción
# -----------------------------
st.title("Predicción de Consumo Eléctrico")
st.markdown("""
Esta app predice el **Consumo Global Activo (kW)** de un hogar
y permite comparar el consumo real vs predicho en un dataset histórico.
""")

# -----------------------------
# 2️⃣ Cargar modelo y dataset de ejemplo
# -----------------------------
@st.cache_data
def cargar_modelo():
    return joblib.load("modelo_xgb_500k.pkl")

@st.cache_data
def cargar_dataset():
    return pd.read_csv("household_power_consumption_sample.csv")

model = cargar_modelo()
df_sample = cargar_dataset()

# -----------------------------
# 3️⃣ Inputs del usuario
# -----------------------------
st.sidebar.header("Características del hogar y tiempo")

Global_reactive_power = st.sidebar.slider("Potencia Reactiva (kW)", 0.0, 1.0, 0.1)
Voltage = st.sidebar.slider("Voltaje (V)", 220.0, 250.0, 235.0)
Global_intensity = st.sidebar.slider("Intensidad Global (A)", 0.0, 30.0, 10.0)
Sub_metering_1 = st.sidebar.slider("Cocina (Wh)", 0, 50, 10)
Sub_metering_2 = st.sidebar.slider("Lavandería (Wh)", 0, 50, 10)
Sub_metering_3 = st.sidebar.slider("Agua Caliente/AC (Wh)", 0, 50, 10)

hour = st.sidebar.slider("Hora del día", 0, 23, 12)
day_of_week = st.sidebar.slider("Día de la semana (0=Lunes,6=Domingo)", 0, 6, 0)
month = st.sidebar.slider("Mes", 1, 12, 1)
is_weekend = 1 if day_of_week >= 5 else 0

GAP_rolling_mean_60 = st.sidebar.slider("Media móvil 60 min (kW)", 0.0, 5.0, 0.1)
GAP_rolling_mean_120 = st.sidebar.slider("Media móvil 120 min (kW)", 0.0, 5.0, 0.1)
GAP_diff_1 = st.sidebar.slider("Diferencia 1 min (kW)", -1.0, 1.0, 0.0)
GAP_diff_60 = st.sidebar.slider("Diferencia 60 min (kW)", -1.0, 1.0, 0.0)
sub_metering_total = Sub_metering_1 + Sub_metering_2 + Sub_metering_3

features = np.array([[Global_reactive_power, Voltage, Global_intensity,
                      Sub_metering_1, Sub_metering_2, Sub_metering_3,
                      hour, day_of_week, month, is_weekend,
                      GAP_rolling_mean_60, GAP_rolling_mean_120,
                      GAP_diff_1, GAP_diff_60, sub_metering_total]])

# -----------------------------
# 4️⃣ Predicción
# -----------------------------
pred = model.predict(features)[0]
st.success(f"Predicción de Consumo Global Activo: {pred:.3f} kW")

# -----------------------------
# 5️⃣ Comparación Real vs Predicho
# -----------------------------
st.subheader("Comparación Real vs Predicho")

feature_cols = ['Global_reactive_power','Voltage','Global_intensity',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3',
                'hour','day_of_week','month','is_weekend',
                'GAP_rolling_mean_60','GAP_rolling_mean_120',
                'GAP_diff_1','GAP_diff_60','sub_metering_total']

X_sample = df_sample[feature_cols]
y_real = df_sample['Global_active_power']
y_pred = model.predict(X_sample)

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_real[:1000].values, label="Real", alpha=0.8)
ax.plot(y_pred[:1000], label="Predicho", alpha=0.8)
ax.set_title("Consumo Real vs Predicho (Primeros 1000 puntos)")
ax.set_xlabel("Tiempo")
ax.set_ylabel("Consumo (kW)")
ax.legend()
st.pyplot(fig)
