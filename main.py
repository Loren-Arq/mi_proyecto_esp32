import os
import io
import time
import wave
import librosa
import requests
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from fastapi import FastAPI, Request, Header, BackgroundTasks
import uvicorn
from scipy.signal import butter, lfilter
from contextlib import asynccontextmanager
import traceback
from Adafruit_IO import Client



# ==============================================================================
# --- 1. CONFIGURACIÓN GLOBAL ---
# ==============================================================================
API_PORT        = int(os.environ.get("PORT", 8080))
SAMPLE_RATE     = 16000
RECORD_SECONDS  = 3
# Esto creará una carpeta llamada "audios_temp" dentro del mismo directorio del proyecto
OUTPUT_DIR = os.path.join(os.getcwd(), "audios_temp")

FACTOR_AMP      = 10.0
MODEL_PATH = 'mi_modelo_aedes.tflite'

# Variables globales para el intérprete TFLite
interpreter    = None
input_details  = None
output_details = None




# 🌐 ADAFRUIT IO — Reemplaza con tus datos reales
import os

# Tu código lee la llave de forma segura desde el panel de Railway
ADAFRUIT_USERNAME = os.getenv("ADAFRUIT_AIO_USERNAME")
ADAFRUIT_KEY = os.getenv("ADAFRUIT_AIO_KEY")


# 📍 Feeds que debes crear en io.adafruit.com con exactamente estos nombres:
#    probabilidad-aedes  /  frecuencia-hz  /  distancia-mm  /  amplitud-db
FEED_PROBABILIDAD = "feed-probabilidad"
FEED_FRECUENCIA   = "feed-frecuencia"
FEED_DISTANCIA    = "feed-distancia"
FEED_AMPLITUD     = "feed-amplitud"
FEED_LATENCIA_RED = "feed-latencia-red"
FEED_LATENCIA_CNN = "feed-latencia-cnn"
FEED_UBICACION    = "sensor-ubicacion"

# 📍 COORDENADAS FIJAS DEL SENSOR
LATITUD  = 14.58849
LONGITUD = -90.55330

# Reporte Excel local
TIMESTAMP_INICIO = datetime.now().strftime("%Y%m%d_%H%M%S")
EXCEL_NAME       = f'Reporte_Aedes_{TIMESTAMP_INICIO}.xlsx'

os.makedirs(OUTPUT_DIR, exist_ok=True)

resultados       = []
contador_evento  = 1
model            = None

# ==============================================================================
# --- 3. FUNCIÓN DE ENVÍO A ADAFRUIT IO ---
# ==============================================================================
def enviar_a_adafruit(feed_key, valor):
    """
    Envía un valor numérico a un feed de Adafruit IO vía REST.
    Se llama en un hilo separado para no bloquear la recepción de audio.
    """
    # 1. Extraemos de forma segura las variables configuradas en Railway
    username = os.getenv("ADAFRUIT_IO_USERNAME")
    aio_key = os.getenv("ADAFRUIT_IO_KEY")
    
    if not username or not aio_key:
        print("⚠️ Error: Faltan las variables de entorno en Railway.")
        return
    
    try:
        # 2. Corregido: Usamos 'username' y 'aio_key' que acabamos de leer arriba
        url = f"https://io.adafruit.com/api/v2/{username}/feeds/{feed_key}/data"
        headers = {
            "X-AIO-Key": aio_key,
            "Content-Type": "application/json"
        }
        payload = {"value": str(valor)}
        response = requests.post(url, json=payload, headers=headers, timeout=5)

        if response.status_code in [200, 201]:
            print(f"  ☁️  [{feed_key}] → {valor}  ✔ enviado a Adafruit IO")
        else:
            print(f"  ⚠️  [{feed_key}] Error Adafruit: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"  ❌ [{feed_key}] No se pudo enviar a Adafruit IO: {e}")

def enviar_ubicacion_a_adafruit(prob):
    """Envia coordenadas con valor al feed del mapa."""
    username = os.getenv("ADAFRUIT_IO_USERNAME")
    aio_key = os.getenv("ADAFRUIT_IO_KEY")
    
    if not username or not aio_key:
        print("⚠️ Error: Faltan las variables de entorno para el mapa en Railway.")
        return

    try:
        # Corregido: Usamos las mismas variables locales de entorno
        url = f"https://io.adafruit.com/api/v2/{username}/feeds/{FEED_UBICACION}/data"
        headers = {
            "X-AIO-Key":    aio_key,
            "Content-Type": "application/json"
        }
        # Formato especial que Adafruit necesita para el mapa
        payload = {
            "value":   str(round(prob * 100, 2)),
            "lat":     LATITUD,
            "lon":     LONGITUD,
            "ele":     0
        }
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code in [200, 201]:
            print(f"  🗺️  [mapa] → ubicación enviada ✔")
        else:
            print(f"  ⚠️  [mapa] Error: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"  ❌ [mapa] Error: {e}")



def enviar_todos_a_adafruit(prob, freq, distancia, amp_db, latencia_red, latencia_cnn):
    """Lanza 4 hilos en paralelo para enviar todos los feeds a la vez."""
    datos = [
        (FEED_PROBABILIDAD, round(prob * 100, 2)),   # como porcentaje numérico
        (FEED_FRECUENCIA,   round(freq, 2)),
        (FEED_DISTANCIA,    distancia),
        (FEED_AMPLITUD,     round(amp_db, 2)),
        (FEED_LATENCIA_RED, latencia_red),
        (FEED_LATENCIA_CNN, latencia_cnn),
    ]
    hilos = []
    for feed, valor in datos:
        h = threading.Thread(target=enviar_a_adafruit, args=(feed, valor))
        h.daemon = True
        h.start()
        hilos.append(h)
        # Hilo especial para el mapa con coordenadas
    h_mapa = threading.Thread(target=enviar_ubicacion_a_adafruit, args=(prob,))
    h_mapa.daemon = True
    h_mapa.start()
    hilos.append(h_mapa)

    # Espera máximo 6 segundos a que terminen todos
    for h in hilos:
        h.join(timeout=6)


# ==============================================================================
# --- 4. FUNCIONES DE AUDIO Y PROCESAMIENTO CNN ---
# ==============================================================================
def filtro_pasa_alta(data, sr):
    cutoff = 300
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        return data
    b, a = butter(6, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)


def procesar_audio_aedes(y, sr):
    y_filtrado = filtro_pasa_alta(y, sr)
    if np.max(np.abs(y_filtrado)) > 0:
        y_norm = librosa.util.normalize(y_filtrado)
    else:
        y_norm = y_filtrado
    return y_norm


def analizar_mosquito(file_path, model):
    try:
        time.sleep(0.05)

        y_raw, sr = librosa.load(file_path, sr=None)

        if len(y_raw) == 0:
            print("⚠️ [Error de Captura] El archivo de audio llegó vacío.")
            return 0.0, 0.0, -80.0, "No detectados"

        rms         = librosa.feature.rms(y=y_raw)
        rms_medio   = np.mean(rms)
        amplitud_db = 20 * np.log10(rms_medio) if rms_medio > 0 else -80.0

        y = procesar_audio_aedes(y_raw, sr)

        S      = np.abs(librosa.stft(y))
        f      = librosa.fft_frequencies(sr=sr)
        S_mean = np.mean(S, axis=1)

        mask = (f >= 200) & (f <= 2000)
        if np.any(mask):
            f_sub          = f[mask]
            S_sub          = S_mean[mask]
            freq_dominante = f_sub[np.argmax(S_sub)]

            armonicos_detectados = []
            for i in [2, 3, 4]:
                target_freq = freq_dominante * i
                if target_freq < (sr / 2):
                    mask_armonico = (f >= (target_freq - 50)) & (f <= (target_freq + 50))
                    if np.any(mask_armonico):
                        f_arm           = f[mask_armonico]
                        S_arm           = S_mean[mask_armonico]
                        freq_real_arm   = f_arm[np.argmax(S_arm)]
                        armonicos_detectados.append(f"{freq_real_arm:.1f} Hz")
                    else:
                        armonicos_detectados.append(f"~{target_freq:.1f} Hz")
                else:
                    armonicos_detectados.append("N/A")
            str_armonicos = " | ".join(armonicos_detectados)
        else:
            freq_dominante = 0.0
            str_armonicos  = "No detectados"

        mel_spec    = librosa.feature.melspectrogram(y=y, sr=sr, fmin=200, fmax=2000, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # --- REEMPLAZADO CON INFERENCIA TFLITE ---
        img = tf.image.resize(mel_spec_db[..., np.newaxis], (128, 128)).numpy()
        if np.max(img) - np.min(img) != 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # Convertimos la matriz a float32 y agregamos la dimensión de lote (Batch) requerida por TFLite
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        # Inyectamos el espectrograma en el índice de entrada de TFLite
        interpreter.set_tensor(input_details['index'], input_data)
        
        # Ejecutamos la predicción matemática compacta
        interpreter.invoke()
        
        # Extraemos el resultado del tensor de salida
        pred = interpreter.get_tensor(output_details['index'])

        if isinstance(pred, np.ndarray):
            probabilidad = float(pred.flatten()[0])
        else:
            probabilidad = float(pred)

        # Tu filtro de validación de frecuencia para el aleteo del Aedes
        if (freq_dominante < 380.0) or (freq_dominante > 620.0):
            probabilidad = 0.0

        return probabilidad, freq_dominante, amplitud_db, str_armonicos

    except Exception as e:
        print(f"\n❌ [ERROR EN MATRIZ DE IA / AUDIO]: {e}")
        return 0.0, 0.0, -80.0, "Error en procesamiento"

# ==============================================================================
# --- 5. PROCESAMIENTO EN SEGUNDO PLANO Y ENVÍO ---
# ==============================================================================

def enviar_todos_a_adafruit(prob, freq, distancia_mm, amp, lat_red, lat_cnn):
    """Función independiente encargada estrictamente de la conexión con Adafruit"""
    username = os.getenv("ADAFRUIT_IO_USERNAME")
    key = os.getenv("ADAFRUIT_IO_KEY")
    
    if not username or not key:
        print("⚠️ Error: Faltan las variables de entorno en Railway. Configúralas en el panel de control.")
        raise ValueError("Credenciales de Adafruit ausentes.")
        
    try:
        # Inicializar el cliente de Adafruit
        aio = Client(username, key)
        
        # Envío de datos con filtros numéricos seguros
        aio.send_data('feed-probabilidad', float(prob))
        aio.send_data('feed-frecuencia', float(freq))
        
        try:
            aio.send_data('feed-distancia', int(distancia_mm))
        except:
            print(f"⚠️ Advertencia: No se pudo enviar distancia válida ('{distancia_mm}')")
        
        print("🚀 Datos enviados a Adafruit IO correctamente.")
        
    except Exception as e:
        print(f"❌ Error físico al conectar o enviar a Adafruit: {e}")
        raise e


def procesar_audio_e_inferencia(raw_audio, distancia_mm, hora_detectada,
                                 timestamp_file, ts_llegada, latencia_red_ms):
    """Lógica pesada en segundo plano — no congela al ESP32"""
    global contador_evento
    
    nombre_archivo = f'audio_{timestamp_file}.wav'
    prob, freq, amp_db, armonicos = 0.0, 0.0, 0.0, "N/A"
    latencia_cnn, latencia_total = 0, 0

    try:
        # 1. Medir inicio de CNN
        ts_inicio_cnn = int(time.time() * 1000)

        # 2. Conversión y amplificación
        samples  = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        samples *= FACTOR_AMP
        samples  = np.clip(samples, -32768, 32767).astype(np.int16)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ruta_wav = os.path.join(OUTPUT_DIR, nombre_archivo)

        # 3. Guardar .wav en disco temporal
        with wave.open(ruta_wav, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(samples.tobytes())

        # 4. Análisis CNN utilizando el intérprete global
        prob, freq, amp_db, armonicos = analizar_mosquito(ruta_wav, interpreter)

        # 5. Calcular latencias
        ts_fin_cnn     = int(time.time() * 1000)
        latencia_cnn   = ts_fin_cnn - ts_inicio_cnn
        latencia_total = latencia_cnn + max(latencia_red_ms, 0)
        
        # 6. Enviar a Adafruit IO llamando a la función externa limpiamente
        print("🚀 Intentando enviar datos a Adafruit IO...")
        enviar_todos_a_adafruit(prob, freq, distancia_mm, amp_db, latencia_red_ms, latencia_cnn)
        print("✅ Envío a Adafruit completado con éxito.")

        # 7. Reporte estético en consola en caso de ÉXITO (Sin Excel local)
        sep = "─" * 65
        print(f"\n{sep}")
        print(f"📊 EVENTO #{contador_evento} PROCESADO CON ÉXITO  [{hora_detectada}]")
        print(f"{sep}")
        print(f"  Archivo Registrado : {nombre_archivo}")
        print(f"  Distancia Objetivo : {distancia_mm} mm")
        print(f"  Frecuencia Alateo  : {freq:.2f} Hz")
        print(f"  Intensidad Sonido  : {amp_db:.2f} dB")
        print(f"  Espectro Armónicos : {armonicos}")
        print(f"  Probabilidad Aedes : {prob:.2%}")
        print(f"  ⏱  Latencia Red    : {latencia_red_ms} ms")
        print(f"  ⏱  Latencia CNN    : {latencia_cnn} ms")
        print(f"  ⏱  Latencia Total  : {latencia_total} ms")
        print(f"  ☁️  Datos enviados a Adafruit IO")
        print(f"{sep}\n")

        contador_evento += 1

    except Exception as e:
        print("💥 ERROR CRÍTICO EN LA TAREA EN SEGUNDO PLANO:")
        traceback.print_exc()

  
# ==============================================================================
# --- 6+. ARRANQUE DEL SERVIDOR ---
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details
    print("🚀 Iniciando Servidor... Cargando motor TFLite.")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Error crítico: No se encontró el archivo '{MODEL_PATH}' en la raíz del proyecto.")
        
    # Inicializar el intérprete TFLite globalmente
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Extraer las dimensiones requeridas de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ Motor TFLite cargado correctamente y listo para Railway.")
    yield
    print("🛑 Servidor apagado.")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def recibir_audio_wifi(request: Request, background_tasks: BackgroundTasks): # 👈 Agregamos 'async'
    try:
        # ⚡️ Ahora leemos el cuerpo de forma asíncrona nativa, ultra rápido:
        raw_audio = await request.body() 
        
        timestamp_llegada_railway = int(time.time() * 1000)
        distancia_mm = request.headers.get("X-Distance", "?")
        ts_esp32_str = request.headers.get("X-Timestamp-ESP32", "0")

        ahora          = datetime.now()
        hora_detectada = ahora.strftime("%H:%M:%S")
        timestamp_file = ahora.strftime("%Y%m%d_%H%M%S")

        try:
            ts_esp32    = int(ts_esp32_str)
            latencia_ms = timestamp_llegada_railway - ts_esp32
            if latencia_ms < 0 or latencia_ms > 60000:
                latencia_ms = -1
        except:
            latencia_ms = -1

        print(f"📡 Audio recibido [{hora_detectada}] — Distancia: {distancia_mm}mm — Latencia red: {latencia_ms}ms")

        # 🧵 Registramos la tarea pesada. FastAPI la procesará en segundo plano
        background_tasks.add_task(
            procesar_audio_e_inferencia,
            raw_audio, distancia_mm, hora_detectada, timestamp_file, timestamp_llegada_railway, latencia_ms
        )
        print("🧵 Tarea de procesamiento registrada con éxito en segundo plano.")

        # 🚀 Esto se le responde al Arduino INMEDIATAMENTE, evitando el Timeout
        return {
            "status":   "recibido",
            "hora":     hora_detectada,
            "latencia": f"{latencia_ms}ms"
        }

    except Exception as e:
        print(f"❌ Error recibiendo petición: {e}")
        return {"status": "error", "message": str(e)}



# ==============================================================================
# --- 7 PUNTO DE ENTRADA ---
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
