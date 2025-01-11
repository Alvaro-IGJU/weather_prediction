import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import locale

# Establecer la configuración regional para español
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')  # Para sistemas Unix
# Para Windows, usa 'Spanish_Spain.1252'
# locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')

# Cargar el modelo entrenado
model = joblib.load('weather_model.pkl')

# Cargar PCA y otras transformaciones si es necesario
try:
    pca = joblib.load('pca_model.pkl')
except FileNotFoundError:
    st.error("El modelo PCA no se encontró. Asegúrate de haberlo guardado y proporcionar la ruta correcta.")
    pca = None

# Cargar las columnas ajustadas durante el entrenamiento
fitted_columns = joblib.load('fitted_columns.pkl')

# Cargar escalador para normalizar los datos
scaler = joblib.load('scaler.pkl')

# Título de la aplicación
st.title("Predicción Meteorológica para los Próximos Días")
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
    'Soleado': "☀️"
}

# Recoger entradas para cada día
days = []
for i in range(num_days):
    st.subheader(f"Día {i + 1}")
    col1, col2 = st.columns(2)
    with col1:
        precipitation = st.number_input(f"Precipitación (mm):", key=f"precipitation_{i}", value=0.0, min_value=0.0, max_value=500.0)
        temp_max = st.number_input(f"Temperatura máxima (°C):", key=f"temp_max_{i}", value=30.0, min_value=-50.0, max_value=60.0)
        temp_min = st.number_input(f"Temperatura mínima (°C):", key=f"temp_min_{i}", value=20.0, min_value=-50.0, max_value=60.0)
        wind = st.number_input(f"Velocidad del viento (km/h):", key=f"wind_{i}", value=10.0, min_value=0.0, max_value=200.0)
        cloudiness_name = st.selectbox(f"Índice de nubosidad:", options=list(cloudiness_map.keys()), key=f"cloudiness_name_{i}")
        cloudiness_id = cloudiness_map[cloudiness_name]
    with col2:
        humidity = st.number_input(f"Humedad relativa (%):", key=f"humidity_{i}", value=50.0, min_value=0.0, max_value=100.0)
        pressure = st.number_input(f"Presión atmosférica (hPa):", key=f"pressure_{i}", value=1013.0, min_value=800.0, max_value=1200.0)
        solar_radiation = st.number_input(f"Radiación solar (W/m²):", key=f"solar_radiation_{i}", value=500.0, min_value=0.0, max_value=2000.0)
        visibility = st.number_input(f"Visibilidad (km):", key=f"visibility_{i}", value=10.0, min_value=0.0, max_value=50.0)
        estacion_name = st.selectbox(f"Estación:", options=list(estacion_map.keys()), key=f"estacion_name_{i}")
        estacion_id = estacion_map[estacion_name]
    days.append([precipitation, temp_max, temp_min, wind, humidity, pressure, solar_radiation, visibility, cloudiness_id, estacion_id])

# Lista de tipos de clima
weather_types = ['Niebla', 'Lluvia', 'Tormenta', 'Soleado']

if st.button("Predecir"):
    if days:
        # Crear DataFrame de entrada
        input_data = pd.DataFrame(days, columns=[
            'precipitation', 'temp_max', 'temp_min', 'wind', 'humidity', 'pressure', 
            'solar_radiation', 'visibility', 'cloudiness_id', 'estacion_id'
        ])

        # Generar dummies para columnas categóricas
        input_data['cloudiness'] = input_data['cloudiness_id'].map({1: 'Parcialmente nublado', 2: 'Despejado', 3: 'Cubierto'})
        input_data['estacion'] = input_data['estacion_id'].map({1: 'Invierno', 2: 'Primavera', 3: 'Verano', 4: 'Otoño'})
        input_data = pd.get_dummies(input_data, columns=['cloudiness', 'estacion'], prefix=['cloudiness', 'estacion'])

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

        # Mostrar los resultados en formato horizontal
        st.write("### Predicciones para los próximos días:")
        cols = st.columns(num_days)  # Crear columnas para los días
        start_date = datetime.now() + timedelta(days=1)  # Comenzar desde mañana
        for day, col in enumerate(cols):
            weather_type = weather_types[predictions[day].argmax()]
            temp_max = int(input_data.iloc[day]['temp_max'])
            temp_min = int(input_data.iloc[day]['temp_min'])
            day_label = (start_date + timedelta(days=day)).strftime('%A').capitalize()  # Nombre del día en español
            with col:
                card_html = f"""
                    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; 
                                box-shadow: 0px 4px 6px rgba(0,0,0,0.1); text-align: center; 
                                width: 150px; height: 250px; margin: 0 auto;">
                        <div style="font-size: 50px; margin-bottom: 10px;">
                            {weather_icons[weather_type]}
                        </div>
                        <div style="font-size: 14px; font-weight: bold; margin-bottom: 10px;">
                            {day_label}
                        </div>
                        <div style="font-size: 12px; color: #555; margin-bottom: 10px;">
                            {weather_type}
                        </div>
                        <div style="font-size: 16px; font-weight: bold; color: #333;">
                            {temp_max}° <span style="color: #888;">{temp_min}°</span>
                        </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
