import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Importar el modelo guardado

# Cargar el modelo entrenado
model = joblib.load('weather_model.pkl')  # Cambia esta ruta según donde guardes tu modelo

# Cargar PCA si es necesario (asegúrate de guardar el PCA en un archivo separado)
try:
    pca = joblib.load('pca_model.pkl')  # Ruta al archivo del PCA
except FileNotFoundError:
    st.error("El modelo PCA no se encontró. Asegúrate de haberlo guardado y proporcionar la ruta correcta.")
    pca = None

# Título de la aplicación
st.title("Predicción Meteorológica para los Próximos Días")
st.markdown("Ingresa las características climáticas para predecir el clima de los próximos días. 😊")

# Número de días a predecir
num_days = st.number_input("Número de días a predecir:", min_value=1, max_value=10, step=1, value=1)

# Recoger entradas para cada día
days = []
for i in range(num_days):
    st.subheader(f"Día {i + 1}")
    precipitation = st.number_input(f"Precipitación (mm):", key=f"precipitation_{i}", value=0.0, min_value=0.0, max_value=500.0)
    temp_max = st.number_input(f"Temperatura máxima (°C):", key=f"temp_max_{i}", value=30.0, min_value=-50.0, max_value=60.0)
    temp_min = st.number_input(f"Temperatura mínima (°C):", key=f"temp_min_{i}", value=20.0, min_value=-50.0, max_value=60.0)
    wind = st.number_input(f"Velocidad del viento (km/h):", key=f"wind_{i}", value=10.0, min_value=0.0, max_value=200.0)
    humidity = st.number_input(f"Humedad relativa (%):", key=f"humidity_{i}", value=50.0, min_value=0.0, max_value=100.0)
    pressure = st.number_input(f"Presión atmosférica (hPa):", key=f"pressure_{i}", value=1013.0, min_value=800.0, max_value=1200.0)
    solar_radiation = st.number_input(f"Radiación solar (W/m²):", key=f"solar_radiation_{i}", value=500.0, min_value=0.0, max_value=2000.0)
    visibility = st.number_input(f"Visibilidad (km):", key=f"visibility_{i}", value=10.0, min_value=0.0, max_value=50.0)
    cloudiness_id = st.selectbox(f"Índice de nubosidad (0-10):", options=list(range(0, 11)), key=f"cloudiness_id_{i}", index=5)
    estacion_id = st.selectbox(f"Estación (ID):", options=[1, 2, 3, 4], key=f"estacion_id_{i}")  # Reemplazar con IDs válidos
    days.append([precipitation, temp_max, temp_min, wind, humidity, pressure, solar_radiation, visibility, cloudiness_id, estacion_id])

# Predicción al presionar el botón
if st.button("Predecir"):
    if days:
        # Crear DataFrame de entrada
        input_data = pd.DataFrame(days, columns=[
            'precipitation', 'temp_max', 'temp_min', 'wind', 'humidity', 'pressure', 
            'solar_radiation', 'visibility', 'cloudiness_id', 'estacion_id'
        ])
        st.write("### Datos de entrada antes del PCA:")
        st.dataframe(input_data)
       
        # Aplicar PCA si está disponible
        if pca is not None:
            try:
                # Convertir a array NumPy antes de aplicar el PCA
                input_data_array = input_data.values  # Convertir DataFrame a array
                input_data_pca = pca.transform(input_data_array)  # Transformar los datos usando PCA
                st.write("### Datos transformados por el PCA:")
                st.dataframe(input_data_pca)
                
                # Predicción del modelo
                predictions = model.predict(input_data_pca)
                st.write("### Predicciones del modelo (sin procesar):")
                st.write(predictions)
                
                # Verificar la forma de las predicciones y construir el DataFrame de salida
                if len(predictions.shape) == 1:
                    prediction_df = pd.DataFrame(predictions, columns=['Predicción'])
                else:
                    num_columns = predictions.shape[1]
                    column_names = [f"Clase_{i+1}" for i in range(num_columns)]
                    prediction_df = pd.DataFrame(predictions, columns=column_names)
                
                # Mostrar resultados
                st.write("### Resultados de las Predicciones:")
                st.dataframe(prediction_df)
            except Exception as e:
                st.error(f"Error al aplicar el PCA o predecir: {str(e)}")
        else:
            st.error("No se puede realizar la predicción porque falta el PCA.")
    else:
        st.warning("Por favor, ingresa datos válidos para predecir.")
