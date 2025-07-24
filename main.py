import requests
import schedule
import time
from telegram import Bot
from datetime import datetime
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import os

# Inicializa el bot de Telegram con variables de entorno
TOKEN = os.getenv("7973912924:AAGkIOYd6DMbTfQyEsDn8a8K8oNMxkgXqPc")
CHAT_ID = os.getenv("-4873547081")
bot = Bot(token=TOKEN)

# Inicializa o carga el modelo
model_path = "model.pkl"
if os.path.exists(model_path):
    model = pd.read_pickle(model_path)
else:
    model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, warm_start=True)

# Historial de entrenamiento
history_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'target'])

def get_data():
    """Obtiene los últimos datos de 1 minuto de EURUSDT desde Binance."""
    url = "https://api.binance.com/api/v3/klines?symbol=EURUSDT&interval=1m&limit=2"
    response = requests.get(url)
    data = response.json()

    # Última vela completa
    last_candle = data[-2]
    open_price = float(last_candle[1])
    high = float(last_candle[2])
    low = float(last_candle[3])
    close = float(last_candle[4])
    volume = float(last_candle[5])
    return open_price, high, low, close, volume

def predict_market():
    global history_df

    try:
        open_p, high, low, close, volume = get_data()
        X_input = np.array([[open_p, high, low, close, volume]])

        # Generar predicción
        prediction = model.predict(X_input)[0]
        direction = "🔼 SUBE" if prediction == 1 else "🔽 BAJA"
        bot.send_message(chat_id=CHAT_ID, text=f"📊 EURUSDT\nPróximos 5 minutos: {direction}")

        # Esperar 5 minutos para comprobar si acertó
        time.sleep(300)
        _, _, _, future_close, _ = get_data()
        actual = 1 if future_close > close else 0

        # Entrenar con el nuevo dato real
        model.partial_fit(X_input, [actual], classes=[0, 1])
        pd.to_pickle(model, model_path)

    except Exception as e:
        bot.send_message(chat_id=CHAT_ID, text=f"⚠️ Error al obtener datos o predecir: {str(e)}")

# Ejecutar cada 10 minutos
schedule.every(10).minutes.do(predict_market)

while True:
    schedule.run_pending()
    time.sleep(1)

