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
from fastapi import FastAPI, Request, Header
import uvicorn
from scipy.signal import butter, lfilter
import gdown

# ==============================================================================
# --- 1. CONFIGURACIÓN GLOBAL ---
# ==============================================================================
API_PORT        = int(os.environ.get("PORT", 8080))
SAMPLE_RATE     = 16000
RECORD_SECONDS  = 3
OUTPUT_DIR      = 'AudiosMayo/Audio'
FACTOR_AMP      = 10.0
MODEL_PATH      = 'mi_modelo_aedes.h5'
GDRIVE_ID = "1nGwZzvRF6HQrS5HhUWK5xzz3hZcqJwIx"

def descargar_modelo_si_no_existe():
    if not os.path.exists(MODEL_PATH):
        print("⬇️  Descargando modelo desde Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            MODEL_PATH,
            quiet=False
        )
        print("✅ Modelo descargado.")

# 🌐 ADAFRUIT IO — Reemplaza con tus datos reales
ADAFRUIT_IO_USERNAME = "Loren60"   # 🔴 Cambia esto
ADAFRUIT_IO_KEY      = "Riv382&&sol"        # 🔴 Cambia esto

# 📍 Feeds que debes crear en io.adafruit.com con exactamente estos nombres:
#    probabilidad-aedes  /  frecuencia-hz  /  distancia-mm  /  amplitud-db
FEED_PROBABILIDAD = "probabilidad-aedes"
FEED_FRECUENCIA   = "frecuencia-hz"
FEED_DISTANCIA    = "distancia-mm"
FEED_AMPLITUD     = "amplitud-db"

# 📍 COORDENADAS FIJAS DEL SENSOR
LATITUD  = 14.5875
LONGITUD = -90.552917

# Reporte Excel local
TIMESTAMP_INICIO = datetime.now().strftime("%Y%m%d_%H%M%S")
EXCEL_NAME       = f'Reporte_Aedes_{TIMESTAMP_INICIO}.xlsx'

os.makedirs(OUTPUT_DIR, exist_ok=True)

resultados       = []
contador_evento  = 1
model            = None


# ==============================================================================
# --- 2. FUNCIÓN DE ENVÍO A ADAFRUIT IO ---
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


def enviar_todos_a_adafruit(prob, freq, distancia, amp_db):
    """Lanza 4 hilos en paralelo para enviar todos los feeds a la vez."""
    datos = [
        (FEED_PROBABILIDAD, round(prob * 100, 2)),   # como porcentaje numérico
        (FEED_FRECUENCIA,   round(freq, 2)),
        (FEED_DISTANCIA,    distancia),
        (FEED_AMPLITUD,     round(amp_db, 2)),
    ]
    hilos = []
    for feed, valor in datos:
        h = threading.Thread(target=enviar_a_adafruit, args=(feed, valor))
        h.daemon = True
        h.start()
        hilos.append(h)
    # Espera máximo 6 segundos a que terminen todos
    for h in hilos:
        h.join(timeout=6)


# ==============================================================================
# --- 3. FUNCIONES DE AUDIO Y PROCESAMIENTO CNN ---
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

        img = tf.image.resize(mel_spec_db[..., np.newaxis], (128, 128)).numpy()
        if np.max(img) - np.min(img) != 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)

        if isinstance(pred, np.ndarray):
            probabilidad = float(pred.flatten()[0])
        else:
            probabilidad = float(pred)

        if (freq_dominante < 380.0) or (freq_dominante > 620.0):
            probabilidad = 0.0

        return probabilidad, freq_dominante, amplitud_db, str_armonicos

    except Exception as e:
        print(f"\n❌ [ERROR EN MATRIZ DE IA / AUDIO]: {e}")
        return 0.0, 0.0, -80.0, "Error en procesamiento"


# ==============================================================================
# --- 4. BACKEND RECEPTOR HTTP (FASTAPI) ---
# ==============================================================================
app = FastAPI()

@app.post("/predict")
async def recibir_audio_wifi(request: Request):
    global contador_evento, resultados, model

    try:
        # 1. Capturar ráfaga binaria y metadatos
        raw_audio    = await request.body()
        distancia_mm = request.headers.get("X-Distance", "?")

        ahora          = datetime.now()
        hora_detectada = ahora.strftime("%H:%M:%S")
        timestamp_file = ahora.strftime("%Y%m%d_%H%M%S")

        # 2. Conversión y amplificación
        samples  = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        samples *= FACTOR_AMP
        samples  = np.clip(samples, -32768, 32767).astype(np.int16)

        nombre_archivo = f'audio_{timestamp_file}.wav'
        ruta_wav       = os.path.join(OUTPUT_DIR, nombre_archivo)

        # 3. Guardar .wav en disco
        with wave.open(ruta_wav, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(samples.tobytes())

        # 4. Análisis CNN
        prob, freq, amp_db, armonicos = analizar_mosquito(ruta_wav, model)

        # 5. ☁️ Enviar los 4 datos a Adafruit IO en paralelo (hilo de fondo)
        hilo_adafruit = threading.Thread(
            target=enviar_todos_a_adafruit,
            args=(prob, freq, distancia_mm, amp_db)
        )
        hilo_adafruit.daemon = True
        hilo_adafruit.start()

        # 6. Guardar en Excel local
        resultados.append({
            'Evento':              contador_evento,
            'Archivo':             nombre_archivo,
            'Distancia (mm)':      distancia_mm,
            'Hora Detectada':      hora_detectada,
            'Probabilidad':        f"{prob:.2%}",
            'Frecuencia Central':  f"{freq:.2f} Hz",
            'Amplitud (dB)':       f"{amp_db:.2f} dB",
            'Armónicos (2x|3x|4x)': armonicos
        })
        pd.DataFrame(resultados).to_excel(EXCEL_NAME, index=False)

        # 7. Reporte en consola
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
        print(f"  ☁️  Enviando datos a Adafruit IO...")
        print(f"{sep}\n")

        contador_evento += 1
        return {
            "status":       "success",
            "evento":       contador_evento - 1,
            "probabilidad": f"{prob:.2%}"
        }

    except Exception as e:
        print(f"❌ Error crítico procesando la petición HTTP: {e}")
        return {"status": "error", "message": str(e)}


# ==============================================================================
# --- 5. ARRANQUE DEL SERVIDOR ---
# ==============================================================================
if __name__ == "__main__":
    descargar_modelo_si_no_existe()   # ← línea nueva
    print("🧠 Cargando modelo de Inteligencia Artificial...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo listo.")
    print(f"📊 Reporte Excel: {EXCEL_NAME}")
    print(f"📡 Esperando señales del ESP32-S3 en el puerto {API_PORT}...\n")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")