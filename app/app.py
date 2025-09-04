# ==============================
# APP STREAMLIT - PREDICCIÓN CONSUMO ENERGÍA
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,5)

# ==============================
# Funciones de carga con caching
# ==============================
@st.cache_resource
def cargar_modelo():
    # Cargar modelo XGBoost desde la misma carpeta
    return joblib.load("modelo_xgb_500k.pkl")

@st.cache_data
def cargar_dataset():
    # Cargar CSV reducido desde la misma carpeta
    return pd.read_csv("household_power_consumption_sample.csv")

# ==============================
# Carga de modelo y dataset
# ==============================
model = cargar_modelo()
df = cargar_dataset()

# ==============================
# Título y descripción
# ==============================
st.title("Predicción de Consumo de Energía - Hogares")
st.markdown("""
Esta aplicación permite predecir el consumo **Global Active Power** en un hogar
utilizando datos históricos y un modelo XGBoost entrenado.
""")

# ==============================
# Visualización del dataset
# ==============================
st.subheader("Vista rápida del dataset")
st.dataframe(df.head())

# ==============================
# Simulación de predicción
# ==============================
st.subheader("Simular predicción")
st.markdown("Selecciona los valores de las variables para predecir el consumo:")

# Inputs interactivos
hour = st.slider("Hora del día", 0, 23, 12)
day_of_week = st.selectbox("Día de la semana (0=Lunes)", range(7), index=0)
month = st.slider("Mes", 1, 12, 6)
is_weekend = 1 if day_of_week >= 5 else 0

global_reactive_power = st.number_input(
    "Global Reactive Power", float(df['Global_reactive_power'].min()),
    float(df['Global_reactive_power'].max()),
    float(df['Global_reactive_power'].mean())
)

voltage = st.number_input(
    "Voltage", float(df['Voltage'].min()),
    float(df['Voltage'].max()),
    float(df['Voltage'].mean())
)

global_intensity = st.number_input(
    "Global Intensity", float(df['Global_intensity'].min()),
    float(df['Global_intensity'].max()),
    float(df['Global_intensity'].mean())
)

sub_metering_1 = st.number_input(
    "Sub Metering 1", float(df['Sub_metering_1'].min()),
    float(df['Sub_metering_1'].max()),
    float(df['Sub_metering_1'].mean())
)

sub_metering_2 = st.number_input(
    "Sub Metering 2", float(df['Sub_metering_2'].min()),
    float(df['Sub_metering_2'].max()),
    float(df['Sub_metering_2'].mean())
)

sub_metering_3 = st.number_input(
    "Sub Metering 3", float(df['Sub_metering_3'].min()),
    float(df['Sub_metering_3'].max()),
    float(df['Sub_metering_3'].mean())
)

# Features derivadas (simplificadas)
GAP_rolling_mean_60 = global_reactive_power
GAP_rolling_mean_120 = global_reactive_power
GAP_diff_1 = global_reactive_power
GAP_diff_60 = global_reactive_power
sub_metering_total = sub_metering_1 + sub_metering_2 + sub_metering_3

X_pred = pd.DataFrame({
    'Global_reactive_power':[global_reactive_power],
    'Voltage':[voltage],
    'Global_intensity':[global_intensity],
    'Sub_metering_1':[sub_metering_1],
    'Sub_metering_2':[sub_metering_2],
    'Sub_metering_3':[sub_metering_3],
    'hour':[hour],
    'day_of_week':[day_of_week],
    'month':[month],
    'is_weekend':[is_weekend],
    'GAP_rolling_mean_60':[GAP_rolling_mean_60],
    'GAP_rolling_mean_120':[GAP_rolling_mean_120],
    'GAP_diff_1':[GAP_diff_1],
    'GAP_diff_60':[GAP_diff_60],
    'sub_metering_total':[sub_metering_total]
})

if st.button("Predecir consumo"):
    pred = model.predict(X_pred)[0]
    st.success(f"Predicción de Global Active Power: {pred:.3f} kW")

# ==============================
# Gráficos de análisis de negocio
# ==============================
st.subheader("Análisis exploratorio")

# Histograma de consumo
fig, ax = plt.subplots()
sns.histplot(df['Global_active_power'], bins=50, kde=True, ax=ax)
ax.set_title("Distribución de Global Active Power")
st.pyplot(fig)

# Consumo promedio por hora
fig2, ax2 = plt.subplots()
df.groupby('hour')['Global_active_power'].mean().plot(kind='bar', ax=ax2)
ax2.set_title("Consumo promedio por hora")
st.pyplot(fig2)
