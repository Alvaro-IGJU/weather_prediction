import streamlit as st
import pandas as pd
import joblib
import base64
from datetime import datetime, timedelta
import locale
import calendar

# Intentar configurar la localización para español
try:
    locale.setlocale(locale.LC_TIME, 'us_US.UTF-8')  # Para sistemas Unix
except locale.Error:
    st.warning("No se pudo configurar la localización para 'us_US.UTF-8'. Asegúrate de que tu sistema soporte esta configuración.")

# Configurar los días de la semana en español
dias_semana = {
    0: 'Lunes',
    1: 'Martes',
    2: 'Miércoles',
    3: 'Jueves',
    4: 'Viernes',
    5: 'Sábado',
    6: 'Domingo'
}

# Cargar el modelo entrenado
try:
    model = joblib.load('pkl_models/weather_model.pkl')
except FileNotFoundError:
    st.error("El modelo no se encontró. Asegúrate de haberlo guardado y proporcionar la ruta correcta.")

# Cargar PCA y otras transformaciones
try:
    pca = joblib.load('pkl_models/pca_model.pkl')
except FileNotFoundError:
    pca = None
    st.warning("El modelo PCA no se encontró. Continuando sin PCA.")

# Cargar las columnas ajustadas durante el entrenamiento
try:
    fitted_columns = joblib.load('pkl_models/fitted_columns.pkl')
except FileNotFoundError:
    fitted_columns = []
    st.error("No se encontraron las columnas ajustadas. Asegúrate de proporcionar el archivo.")

# Cargar el escalador para normalizar los datos
try:
    scaler = joblib.load('pkl_models/scaler.pkl')
except FileNotFoundError:
    scaler = None
    st.error("No se encontró el escalador. Asegúrate de proporcionar el archivo.")

# Ruta del GIF
gif_path = "Izu.gif"

# Leer el GIF y convertirlo en base64
try:
    with open(gif_path, "rb") as gif_file:
        gif_base64 = base64.b64encode(gif_file.read()).decode("utf-8")
except FileNotFoundError:
    gif_base64 = ""
    st.error(f"No se encontró el archivo {gif_path}. Asegúrate de proporcionar la ruta correcta.")

# Estilo personalizado para la página
page_bg = f"""
    <style>
        /* Fondo con el GIF */
        [data-testid="stAppViewContainer"] {{
            background: url(data:image/gif;base64,{gif_base64}) no-repeat center center fixed;
            background-size: cover;
            color: #003366; /* Azul marino */
        }}
        /* Contenedor principal */
        [data-testid="stAppViewContainer"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
            max-width: 90%;
            margin: 20px auto;
        }}
        /* Títulos */
        h1 {{
            color: #003366;
            text-align: center;
            font-weight: bold;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }}
        h3 {{
            color: #0056b3;
            text-align: center;
        }}
        /* Botón estilizado */
        .stButton button {{
            background-color: rgb(21, 16, 173);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
        }}
        .stButton button:hover {{
            background-color: #0056b3;
            transform: scale(1.05);
        }}
        /* Tarjetas */
        .card {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
            margin: 10px auto;
            width: 160px;
        }}
        .card .icon {{
            font-size: 50px;
            color: rgb(21, 16, 173);
            margin-bottom: 10px;
        }}
        .card .day {{
            font-size: 16px;
            font-weight: bold;
            color: rgb(21, 16, 173);
        }}
        .card .prediction {{
            font-size: 14px;
            color: #333;
        }}
    </style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Agregar el nuevo estilo para los inputs con mayor transparencia
st.markdown("""
    <style>
        /* Cambiar fondo de los inputs */
        .stNumberInput input {
            background-color: rgba(200, 200, 200, 0.6) !important;  /* Gris claro con más transparencia */
            border-radius: 5px;
            color: black;
        }
        /* Cambiar el borde de los inputs */
        .stNumberInput input:focus {
            border: 2px solid rgb(21, 16, 173) !important;  /* Borde azul cuando se selecciona */
        }
    </style>
""", unsafe_allow_html=True)


# Título de la aplicación
st.markdown("<h1 style='color: rgb(21, 16, 173); font-size: 54px; font-weight: bold; text-align: center;'>🌤️ MeteoVision</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: black; font-weight: bold; text-align: center;'>Ingresa las características climáticas para predecir el clima de los próximos días. 😊</h4>", unsafe_allow_html=True)

# Número de días a predecir
st.markdown("<h5 style='color: black; font-weight: bold; margin-bottom: -8px;'>Número de días a predecir:</h5>", unsafe_allow_html=True)
num_days = st.number_input("", min_value=1, max_value=10, step=1, value=1)

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
    st.markdown(f"<h3 style='color:rgb(21, 16, 173); font-weight: bold;'>Día {i + 1}</h3>", unsafe_allow_html=True)
    
    # Estilo personalizado para reducir el espacio entre los títulos y los inputs
    st.markdown(""" 
        <style>
            .stNumberInput label {
                margin-top: -10px;  /* Reducción del margen superior */
                margin-bottom: -20px; /* Reducción del margen inferior */
            }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<strong style='color: black; font-weight: bold; font-size: 20px'>Precipitación (mm):</strong>", unsafe_allow_html=True)
        precipitation = st.number_input(
            "", key=f"precipitation_{i}", value=0.0, min_value=0.0, max_value=500.0
        )
        
        st.markdown("<strong style='color: black; font-weight: bold; font-size: 20px'>Temperatura máxima (°C):</strong>", unsafe_allow_html=True)
        temp_max = st.number_input(
            "", key=f"temp_max_{i}", value=30.0, min_value=-50.0, max_value=60.0
        )
        
    with col2:
        st.markdown("<strong style='color: black; font-weight: bold; font-size: 20px'>Velocidad del viento (km/h):</strong>", unsafe_allow_html=True)
        wind = st.number_input(
            "", key=f"wind_{i}", value=10.0, min_value=0.0, max_value=200.0
        )
        
        st.markdown("<strong style='color: black; font-weight: bold; font-size: 20px'>Humedad relativa (%):</strong>", unsafe_allow_html=True)
        humidity = st.number_input(
            "", key=f"humidity_{i}", value=50.0, min_value=0.0, max_value=100.0
        )
    
    days.append([precipitation, temp_max, wind, humidity])

# Estilo para mover el botón a la derecha
st.markdown("""
    <style>
        .stButton > button {
            float: right;
        }
    </style>
""", unsafe_allow_html=True)

# Botón de predicción
if st.button("Predecir"):
    if days and model and scaler:
        input_data = pd.DataFrame(days, columns=['precipitation', 'temp_max', 'wind', 'humidity'])
        for col in fitted_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[fitted_columns]
        input_data_scaled = scaler.transform(input_data)
        if pca:
            input_data_pca = pca.transform(input_data_scaled)
            predictions = model.predict(input_data_pca)
        else:
            predictions = model.predict(input_data_scaled)
        predictions = predictions.argmax(axis=1) if len(predictions.shape) > 1 else predictions

        st.write("<h3 style='color: rgb(21, 16, 173); font-weight: bold;'>Predicciones para los próximos días:</h3>", unsafe_allow_html=True)
        
        # Determinar el número de filas necesarias dependiendo del número de días
        cols_per_row = 5  # Esto es para organizar las predicciones en varias filas si es necesario
        num_rows = (num_days // cols_per_row) + (1 if num_days % cols_per_row > 0 else 0)
        
        start_date = datetime.now() + timedelta(days=1)
        
        # Organizar en filas
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                day = row * cols_per_row + col
                if day < num_days:
                    weather_prediction = weather_types[predictions[day]]
                    weather_icon = weather_icons.get(weather_prediction, "❓")
                    day_label = dias_semana[(start_date + timedelta(days=day)).weekday()]
                    with cols[col]:
                        st.markdown(f"""
                        <div class="card">
                            <div class="icon">{weather_icon}</div>
                            <div class="day">{day_label}</div>
                            <div class="prediction">{weather_prediction}</div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Faltan datos o recursos necesarios para realizar la predicción.")
