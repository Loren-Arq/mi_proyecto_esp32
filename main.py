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

# ==============================================================================
# --- 1. CONFIGURACIÓN GLOBAL ---
# ==============================================================================
API_PORT        = int(os.environ.get("PORT", 8080))
SAMPLE_RATE     = 16000
RECORD_SECONDS  = 3
OUTPUT_DIR      = 'AudiosMayo/Audio'
FACTOR_AMP      = 10.0
MODEL_PATH = 'mi_modelo_aedes.tflite'

# Variables globales para el intérprete TFLite
interpreter    = None
input_details  = None
output_details = None




# 🌐 ADAFRUIT IO — Reemplaza con tus datos reales
ADAFRUIT_IO_USERNAME = "Loren60"   # 🔴 Cambia esto
ADAFRUIT_IO_KEY      = "Riv382&&sol"        # 🔴 Cambia esto

# 📍 Feeds que debes crear en io.adafruit.com con exactamente estos nombres:
#    probabilidad-aedes  /  frecuencia-hz  /  distancia-mm  /  amplitud-db
FEED_PROBABILIDAD = "probabilidad-aedes"
FEED_FRECUENCIA   = "frecuencia-hz"
FEED_DISTANCIA    = "distancia-mm"
FEED_AMPLITUD     = "amplitud-db"
FEED_LATENCIA_RED = "latencia-red-ms"
FEED_LATENCIA_CNN = "latencia-cnn-ms"
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
    try:
        url = f"https://io.adafruit.com/api/v2/{ADAFRUIT_IO_USERNAME}/feeds/{feed_key}/data"
        headers = {
            "X-AIO-Key": ADAFRUIT_IO_KEY,
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
    try:
        url = f"https://io.adafruit.com/api/v2/{ADAFRUIT_IO_USERNAME}/feeds/{FEED_UBICACION}/data"
        headers = {
            "X-AIO-Key":    ADAFRUIT_IO_KEY,
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
            print(f"  ⚠️  [mapa] Error: {response.status_code}")
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
# --- 5. PROCESAMIENTO EN SEGUNDO PLANO ---
# ==============================================================================

def procesar_audio_e_inferencia(raw_audio, distancia_mm, hora_detectada,
                                 timestamp_file, ts_llegada, latencia_red_ms):
    """Lógica pesada en segundo plano — no congela al ESP32"""
    global contador_evento, resultados
    try:
        # 1. Medir inicio de CNN
        ts_inicio_cnn = int(time.time() * 1000)

        # 2. Conversión y amplificación
        samples  = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        samples *= FACTOR_AMP
        samples  = np.clip(samples, -32768, 32767).astype(np.int16)

        nombre_archivo = f'audio_{timestamp_file}.wav'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ruta_wav = os.path.join(OUTPUT_DIR, nombre_archivo)

        # 3. Guardar .wav en disco
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

        # 6. Enviar a Adafruit IO con latencias
        hilo_adafruit = threading.Thread(
            target=enviar_todos_a_adafruit,
            args=(prob, freq, distancia_mm, amp_db, latencia_red_ms, latencia_cnn)
        )
        hilo_adafruit.daemon = True
        hilo_adafruit.start()

        # 7. Guardar en Excel local
        resultados.append({
            'Evento':                contador_evento,
            'Archivo':               nombre_archivo,
            'Distancia (mm)':        distancia_mm,
            'Hora Detectada':        hora_detectada,
            'Probabilidad':          f"{prob:.2%}",
            'Frecuencia Central':    f"{freq:.2f} Hz",
            'Amplitud (dB)':         f"{amp_db:.2f} dB",
            'Armónicos (2x|3x|4x)':  armonicos,
            'Latencia Red (ms)':     latencia_red_ms,
            'Latencia CNN (ms)':     latencia_cnn,
            'Latencia Total (ms)':   latencia_total
        })
        pd.DataFrame(resultados).to_excel(EXCEL_NAME, index=False)

        # 8. Reporte en consola
        sep = "─" * 65
        print(f"\n{sep}")
        print(f"📊 EVENTO #{contador_evento} PROCESADO  [{hora_detectada}]")
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
        print(f"❌ Error en procesamiento de fondo: {e}")
  
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
@app.post("/predict/")
async def recibir_audio_wifi(request: Request, background_tasks: BackgroundTasks):
    try:
        timestamp_llegada_railway = int(time.time() * 1000)
        raw_audio    = await request.body()
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

        print(f"📡 Audio recibido [{hora_detectada}] — "
              f"Distancia: {distancia_mm}mm — "
              f"Latencia red: {latencia_ms}ms")

        background_tasks.add_task(
            procesar_audio_e_inferencia,
            raw_audio,
            distancia_mm,
            hora_detectada,
            timestamp_file,
            timestamp_llegada_railway,
            latencia_ms
        )

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
