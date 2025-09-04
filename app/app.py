import streamlit as st
import joblib
import requests
import io
import numpy as np

st.set_page_config(page_title="Predicción Consumo Energía", layout="centered")

st.title("Predicción de Consumo de Energía")
st.write("Introduce los valores para predecir Global Active Power:")

# ------------------------------
# Función para cargar modelo desde GitHub
# ------------------------------
@st.cache_data(show_spinner=True)
def cargar_modelo():
    url = "https://raw.githubusercontent.com/Maheferrer98/mi_app_streamlit/main/app/modelo_xgb_500k.pkl"
    try:
        response = requests.get(url)
        response.raise_for_status()  # lanzar error si falla
        modelo = joblib.load(io.BytesIO(response.content))
        return modelo
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde GitHub: {e}")
        return None

# ------------------------------
# Cargar modelo
# ------------------------------
model = cargar_modelo()
if model:
    st.success("Modelo cargado correctamente ✅")
else:
    st.warning("No se pudo cargar el modelo ⚠️")

# ------------------------------
# Inputs del usuario
# ------------------------------
hour = st.slider("Hora del día", 0, 23, 12)
day_of_week = st.slider("Día de la semana (0=Lunes, 6=Domingo)", 0, 6, 2)
month = st.slider("Mes (1-12)", 1, 12, 6)
is_weekend = st.selectbox("Es fin de semana?", [0,1], index=0)

global_reactive_power = st.number_input("Global Reactive Power (normalizado)", value=0.0, format="%.4f")
voltage = st.number_input("Voltage (normalizado)", value=0.0, format="%.4f")
global_intensity = st.number_input("Global Intensity (normalizado)", value=0.0, format="%.4f")
sub_metering_1 = st.number_input("Sub Metering 1 (normalizado)", value=0.0, format="%.4f")
sub_metering_2 = st.number_input("Sub Metering 2 (normalizado)", value=0.0, format="%.4f")
sub_metering_3 = st.number_input("Sub Metering 3 (normalizado)", value=0.0, format="%.4f")
GAP_rolling_mean_60 = st.number_input("Rolling mean 60 (normalizado)", value=0.0, format="%.4f")
GAP_rolling_mean_120 = st.number_input("Rolling mean 120 (normalizado)", value=0.0, format="%.4f")
GAP_diff_1 = st.number_input("Diff 1 (normalizado)", value=0.0, format="%.4f")
GAP_diff_60 = st.number_input("Diff 60 (normalizado)", value=0.0, format="%.4f")
sub_metering_total = st.number_input("Submetering total (normalizado)", value=0.0, format="%.4f")

# ------------------------------
# Botón para predecir
# ------------------------------
if st.button("Predecir"):
    if model:
        X_input = np.array([[global_reactive_power, voltage, global_intensity,
                             sub_metering_1, sub_metering_2, sub_metering_3,
                             hour, day_of_week, month, is_weekend,
                             GAP_rolling_mean_60, GAP_rolling_mean_120,
                             GAP_diff_1, GAP_diff_60, sub_metering_total]])
        
        pred = model.predict(X_input)[0]
        st.success(f"Predicción de Global Active Power: {pred:.4f} kW")
    else:
        st.error("El modelo no está cargado, no se puede predecir.")
