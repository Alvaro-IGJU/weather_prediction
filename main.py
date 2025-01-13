import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import locale

# Establecer la configuración regional para español
try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')  # Para sistemas Unix
except locale.Error:
    st.warning("No se pudo configurar la localización para 'en_US.UTF-8'. Asegúrate de que tu sistema soporte esta configuración.")

# Cargar el modelo entrenado
model = joblib.load('pkl_models/weather_model.pkl')

# Cargar PCA y otras transformaciones si es necesario
try:
    pca = joblib.load('pkl_models/pca_model.pkl')
except FileNotFoundError:
    st.error("El modelo PCA no se encontró. Asegúrate de haberlo guardado y proporcionar la ruta correcta.")
    pca = None

# Cargar las columnas ajustadas durante el entrenamiento
fitted_columns = joblib.load('pkl_models/fitted_columns.pkl')

# Cargar escalador para normalizar los datos
scaler = joblib.load('pkl_models/scaler.pkl')

# Título de la aplicación
st.title("MeteoVision")
st.markdown("Ingresa las características climáticas para predecir el clima de los próximos días. 😊")

# Número de días a predecir
num_days = st.number_input("Número de días a predecir:", min_value=1, max_value=10, step=1, value=1)

# Mapeo de estaciones por nombre
estacion_map = {"Invierno": 1, "Primavera": 2, "Verano": 3, "Otoño": 4}

# Mapeo de índice de nubosidad por nombre
cloudiness_map = {"Parcialmente nublado": 1, "Despejado": 2, "Cubierto": 3}

# Diccionario de íconos para el clima
weather_icons = {
    'Niebla': "🌫️",
    'Lluvia': "🌧️",
    'Tormenta': "⛈️",
    'Soleado': "☀️",
    'Nublado': "☁️"
}

# Lista de tipos de clima
weather_types = ['Niebla', 'Lluvia', 'Tormenta', 'Soleado', 'Nublado']

# Recoger entradas para cada día
days = []
for i in range(num_days):
    st.subheader(f"Día {i + 1}")
    col1, col2 = st.columns(2)
    with col1:
        precipitation = st.number_input(f"Precipitación (mm):", key=f"precipitation_{i}", value=0.0, min_value=0.0, max_value=500.0)
        temp_max = st.number_input(f"Temperatura máxima (°C):", key=f"temp_max_{i}", value=30.0, min_value=-50.0, max_value=60.0)
    with col2:
        wind = st.number_input(f"Velocidad del viento (km/h):", key=f"wind_{i}", value=10.0, min_value=0.0, max_value=200.0)
        humidity = st.number_input(f"Humedad relativa (%):", key=f"humidity_{i}", value=50.0, min_value=0.0, max_value=100.0)
    days.append([precipitation, temp_max, wind, humidity])

if st.button("Predecir"):
    if days:
        # Crear DataFrame de entrada
        input_data = pd.DataFrame(days, columns=['precipitation', 'temp_max', 'wind', 'humidity'])

        # Asegurarse de que las columnas coincidan con las usadas en el entrenamiento
        for col in fitted_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[fitted_columns]

        # Normalizar los datos y aplicar PCA si está disponible
        input_data_scaled = scaler.transform(input_data)
        if pca:
            input_data_pca = pca.transform(input_data_scaled)
            predictions = model.predict(input_data_pca)
        else:
            predictions = model.predict(input_data_scaled)

        # Manejar predicciones vacías
        st.write("### Predicciones para los próximos días:")
        cols = st.columns(num_days)
        start_date = datetime.now() + timedelta(days=1)
        for day, col in enumerate(cols):
            prediction_probs = predictions[day]
            weather_prediction = [
                weather_types[i] for i, val in enumerate(prediction_probs) if val == 1
            ]
            if not weather_prediction:
                weather_prediction = ["Nublado"] 
            weather_icons_list = "".join([weather_icons.get(wp, "❓") for wp in weather_prediction])
            day_label = (start_date + timedelta(days=day)).strftime('%A').capitalize()
            with col:
                card_html = f"""
                    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; 
                                box-shadow: 0px 4px 6px rgba(0,0,0,0.1); text-align: center; 
                                width: 150px; height: 250px; margin: 0 auto;">
                        <div style="font-size: 50px; margin-bottom: 10px;">
                            {weather_icons_list}
                        </div>
                        <div style="font-size: 14px; font-weight: bold; margin-bottom: 10px; color: black;">
                            {day_label}
                        </div>
                        <div style="font-size: 12px; color: #555; margin-bottom: 10px;">
                            {", ".join(weather_prediction)}
                        </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
