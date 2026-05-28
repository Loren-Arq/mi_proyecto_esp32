import os
import io
import time
import wave
import librosa
import requests
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from scipy.signal import butter, lfilter
from contextlib import asynccontextmanager
import traceback
import openpyxl
import base64
from github import Github
from pathlib import Path
from io import BytesIO

# ==============================================================================
# --- 1. CONFIGURACIÓN GLOBAL ---
# ==============================================================================
API_PORT       = int(os.environ.get("PORT", 8080))
SAMPLE_RATE    = 16000
RECORD_SECONDS = 3
OUTPUT_DIR     = os.path.join(os.getcwd(), "audios_temp")
FACTOR_AMP     = 10.0
MODEL_PATH     = 'mi_modelo_aedes.tflite'

# Variables globales para el intérprete TFLite
interpreter    = None
input_details  = None
output_details = None

# 📍 Feeds de Adafruit IO
FEED_PROBABILIDAD = "feed-probabilidad"
FEED_FRECUENCIA   = "feed-frecuencia"
FEED_DISTANCIA    = "feed-distancia"
FEED_AMPLITUD     = "feed-amplitud"
FEED_LATENCIA_RED = "feed-latencia-red"
FEED_LATENCIA_CNN = "feed-latencia-cnn"
FEED_UBICACION    = "feed-ubicacion"

# 📍 COORDENADAS FIJAS DEL SENSOR
LATITUD  = 14.58849
LONGITUD = -90.55330

os.makedirs(OUTPUT_DIR, exist_ok=True)

contador_evento = 1

# ==============================================================================
# --- 2. FUNCIONES DE ENVÍO A ADAFRUIT IO ---
# ==============================================================================
def enviar_a_adafruit(feed_key, valor):
    """Envía un valor numérico a un feed de Adafruit IO vía REST."""
    username = os.getenv("ADAFRUIT_IO_USERNAME")
    aio_key  = os.getenv("ADAFRUIT_IO_KEY")

    if not username or not aio_key:
        print("⚠️ Error: Faltan ADAFRUIT_IO_USERNAME o ADAFRUIT_IO_KEY en Railway.")
        return

    try:
        url     = f"https://io.adafruit.com/api/v2/{username}/feeds/{feed_key}/data"
        headers = {"X-AIO-Key": aio_key, "Content-Type": "application/json"}
        payload = {"value": str(valor)}
        response = requests.post(url, json=payload, headers=headers, timeout=5)

        if response.status_code in [200, 201]:
            print(f"  ☁️  [{feed_key}] → {valor}  ✔")
        else:
            print(f"  ⚠️  [{feed_key}] Error Adafruit: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"  ❌ [{feed_key}] No se pudo enviar: {e}")


def enviar_ubicacion_a_adafruit(prob):
    """Envía coordenadas GPS con probabilidad al feed del mapa."""
    username = os.getenv("ADAFRUIT_IO_USERNAME")
    aio_key  = os.getenv("ADAFRUIT_IO_KEY")

    if not username or not aio_key:
        print("⚠️ Error: Faltan variables de entorno para el mapa.")
        return

    try:
        url     = f"https://io.adafruit.com/api/v2/{username}/feeds/{FEED_UBICACION}/data"
        headers = {"X-AIO-Key": aio_key, "Content-Type": "application/json"}
        payload = {
            "value": str(round(prob * 100, 2)),
            "lat":   LATITUD,
            "lon":   LONGITUD,
            "ele":   0
        }
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code in [200, 201]:
            print(f"  🗺️  [ubicacion] → enviada ✔")
        else:
            print(f"  ⚠️  [ubicacion] Error: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"  ❌ [ubicacion] Error: {e}")


def enviar_todos_a_adafruit(prob, freq, distancia, amp_db, latencia_red, latencia_cnn):
    """Lanza hilos en paralelo para enviar todos los feeds a la vez."""
    datos = [
        (FEED_PROBABILIDAD, round(prob * 100, 2)),
        (FEED_FRECUENCIA,   round(freq, 2)),
        (FEED_AMPLITUD,     round(amp_db, 2)),
        (FEED_LATENCIA_RED, latencia_red),
        (FEED_LATENCIA_CNN, latencia_cnn),
    ]

    # Distancia de forma segura
    try:
        datos.append((FEED_DISTANCIA, int(distancia)))
    except:
        print(f"⚠️ Distancia '{distancia}' no es un número válido.")

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

    for h in hilos:
        h.join(timeout=6)

    print("🚀 Envíos a Adafruit IO finalizados.")

 
 # ==============================================================================
# --- 2B. EXCEL EN GITHUB ---
# ==============================================================================
import requests as req_github

GITHUB_TOKEN = os.getenv("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")
GITHUB_USER  = os.getenv("Loren-Arq")   # tu usuario de GitHub
GITHUB_REPO  = os.getenv("mi_proyecto_esp32")   # nombre del repositorio
GITHUB_PATH  = "datos/registros_aedes.xlsx"  # ruta dentro del repo

EXCEL_HEADERS = [
    "Evento", "Fecha", "Hora", "Distancia (mm)",
    "Frecuencia (Hz)", "Amplitud (dB)", "Probabilidad (%)",
    "Armónicos", "Latencia Red (ms)", "Latencia CNN (ms)", "Alerta"
]


def guardar_en_excel_local(fila: list):
   
    """Descarga el Excel de GitHub, agrega la fila y lo vuelve a subir."""
    if not GITHUB_TOKEN or not GITHUB_USER or not GITHUB_REPO:
        raise ValueError("❌ Faltan variables GitHub.")

    try:
        headers_gh = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{GITHUB_PATH}"

        # 1. Intentar descargar el archivo existente
        response = req_github.get(url, headers=headers_gh)
        sha = None

        if response.status_code == 200:
            data      = response.json()
            sha       = data["sha"]  # necesario para actualizar
            contenido = base64.b64decode(data["content"])
            wb        = openpyxl.load_workbook(BytesIO(contenido))
            ws        = wb.active
            print("  📥 Excel descargado de GitHub.")
        else:
            # Crear nuevo Excel si no existe
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Registros Aedes"
            ws.append(EXCEL_HEADERS)
            print("  📄 Excel nuevo creado.")

        # 2. Agregar la fila nueva
        ws.append(fila)

        # 3. Convertir a bytes y subir a GitHub
        buffer = BytesIO()
        wb.save(buffer)
        contenido_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "message": f"Evento #{fila[0]} registrado",
            "content": contenido_b64
        }
        if sha:
            payload["sha"] = sha  # si el archivo ya existía, sha es obligatorio

        put_response = req_github.put(url, headers=headers_gh, json=payload)

        if put_response.status_code in [200, 201]:
            print(f"  ✅ Excel guardado en GitHub: {GITHUB_PATH}")
        else:
            print(f"  ❌ GitHub respondió {put_response.status_code}: {put_response.text}")

    except Exception as e:
        print(f"  ❌ Error al guardar Excel en GitHub: {e}")
        traceback.print_exc()


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
        return librosa.util.normalize(y_filtrado)
    return y_filtrado


def analizar_mosquito(file_path, model=None):
    try:
        time.sleep(0.05)

        y_raw, sr = librosa.load(file_path, sr=None)

        if len(y_raw) == 0:
            print("⚠️ El archivo de audio llegó vacío.")
            return 0.0, 0.0, -80.0, "No detectados"

        # Amplitud
        rms         = librosa.feature.rms(y=y_raw)
        rms_medio   = np.mean(rms)
        amplitud_db = 20 * np.log10(rms_medio) if rms_medio > 0 else -80.0

        y = procesar_audio_aedes(y_raw, sr)

        # Frecuencia dominante y armónicos
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
                    mask_arm = (f >= (target_freq - 50)) & (f <= (target_freq + 50))
                    if np.any(mask_arm):
                        freq_real = f[mask_arm][np.argmax(S_mean[mask_arm])]
                        armonicos_detectados.append(f"{freq_real:.1f} Hz")
                    else:
                        armonicos_detectados.append(f"~{target_freq:.1f} Hz")
                else:
                    armonicos_detectados.append("N/A")
            str_armonicos = " | ".join(armonicos_detectados)
        else:
            freq_dominante = 0.0
            str_armonicos  = "No detectados"

        # Espectrograma mel → TFLite
        mel_spec    = librosa.feature.melspectrogram(y=y, sr=sr, fmin=200, fmax=2000, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        img = tf.image.resize(mel_spec_db[..., np.newaxis], (128, 128)).numpy()
        if (np.max(img) - np.min(img)) != 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        # ✅ Orden correcto: set_tensor → invoke → get_tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        probabilidad = float(pred.flatten()[0]) if isinstance(pred, np.ndarray) else float(pred)

        # Filtro de frecuencia del aleteo del Aedes
        if freq_dominante < 380.0 or freq_dominante > 620.0:
            probabilidad = 0.0

        return probabilidad, freq_dominante, amplitud_db, str_armonicos

    except Exception as e:
        print(f"\n❌ [ERROR EN IA / AUDIO]: {e}")
        return 0.0, 0.0, -80.0, "Error en procesamiento"


# ==============================================================================
# --- 4. PROCESAMIENTO EN SEGUNDO PLANO ---
# ==============================================================================
def procesar_audio_e_inferencia(raw_audio, distancia_mm, hora_detectada,
                                timestamp_file, ts_llegada, latencia_red_ms):
    global contador_evento
    ahora = datetime.now()
    nombre_archivo = f'audio_{timestamp_file}.wav'
    prob, freq, amp_db, armonicos = 0.0, 0.0, 0.0, "N/A"
    latencia_cnn = 0

    try:
        ts_inicio_cnn = int(time.time() * 1000)

        # Conversión y amplificación
        samples  = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        samples *= FACTOR_AMP
        samples  = np.clip(samples, -32768, 32767).astype(np.int16)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ruta_wav = os.path.join(OUTPUT_DIR, nombre_archivo)

        # Guardar .wav temporal
        with wave.open(ruta_wav, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(samples.tobytes())

        # Inferencia CNN
        prob, freq, amp_db, armonicos = analizar_mosquito(ruta_wav)

        ts_fin_cnn     = int(time.time() * 1000)
        latencia_cnn   = ts_fin_cnn - ts_inicio_cnn
        latencia_total = latencia_cnn + max(latencia_red_ms, 0)

        # Enviar a Adafruit
        print("🚀 Enviando datos a Adafruit IO...")
        enviar_todos_a_adafruit(prob, freq, distancia_mm, amp_db, latencia_red_ms, latencia_cnn)
        # 🚨 Alarma si probabilidad > 75%
        #enviar_alarma_adafruit(prob, freq, distancia_mm)

        # 📊 Preparar fila
        alerta = "🚨 SÍ" if prob > 0.75 else "No"
        fila = [
            contador_evento,
            ahora.strftime("%Y-%m-%d"),
            hora_detectada,
            distancia_mm,
            round(freq, 2),
            round(amp_db, 2),
            round(prob * 100, 2),
            armonicos,
            latencia_red_ms,
            latencia_cnn,
            alerta
        ]

        # Reporte en terminal
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
        print(f"{sep}\n")

        # 📊 Guardar en Excel en GitHub
        try:
            guardar_en_excel_local(fila)
            print(f"✅ Excel guardado correctamente en GitHub")
        except Exception as e:
            print(f"❌ Error guardando Excel en GitHub: {type(e).__name__}: {e}")
            traceback.print_exc()

        # Limpiar archivo temporal
        if os.path.exists(ruta_wav):
            os.remove(ruta_wav)

        contador_evento += 1

    except Exception as e:
        print("💥 ERROR CRÍTICO EN SEGUNDO PLANO:")
        traceback.print_exc()

# ==============================================================================
# --- 5. SERVIDOR FASTAPI ---
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details
    print("🚀 Iniciando Servidor... Cargando motor TFLite.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ No se encontró '{MODEL_PATH}'.")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ Motor TFLite listo.")
    yield
    print("🛑 Servidor apagado.")


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def recibir_audio_wifi(request: Request, background_tasks: BackgroundTasks):
    try:
        raw_audio = await request.body()

        timestamp_llegada = int(time.time() * 1000)
        distancia_mm      = request.headers.get("X-Distance", "?")
        latencia_audio    =request.headers.get("X-Latency-Audio-MS") 

        ahora          = datetime.now()
        hora_detectada = ahora.strftime("%H:%M:%S")
        timestamp_file = ahora.strftime("%Y%m%d_%H%M%S")

        # Lee la latencia directamente desde el ESP32
        try:
            latencia_ms = int(latencia_audio)
        except:
            latencia_ms = -1

        print(f"📡 Audio recibido [{hora_detectada}] — Distancia: {distancia_mm}mm — Latencia: {latencia_ms}ms")

        background_tasks.add_task(
            procesar_audio_e_inferencia,
            raw_audio, distancia_mm, hora_detectada,
            timestamp_file, timestamp_llegada, latencia_ms
        )

        return {"status": "recibido", "hora": hora_detectada, "latencia": f"{latencia_ms}ms"}

    except Exception as e:
        print(f"❌ Error en endpoint: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=puerto)
