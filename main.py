import os
import io
from io import BytesIO
import time
import pytz
import wave
import librosa
import requests
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
import requests as req_github
from pathlib import Path

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
contador_evento = 1

# ==============================================================================
# --- 2. CONFIGURACIÓN EXCEL EN GITHUB ---
# ==============================================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USER  = os.getenv("GITHUB_USER")   
GITHUB_REPO  = os.getenv("GITHUB_REPO")   
GITHUB_PATH_BASE  = os.getenv("GITHUB_PATH_BASE", "datos/excel")

print("DEBUG TOKEN:", GITHUB_TOKEN)
print("DEBUG USER :", GITHUB_USER)
print("DEBUG REPO :", GITHUB_REPO)

if not all([GITHUB_TOKEN, GITHUB_USER, GITHUB_REPO]):
    raise ValueError("❌ Faltan variables GitHub en el entorno del servidor.")

EXCEL_HEADERS = [
    "Evento", "Fecha", "Hora", "Distancia (mm)",
    "Frecuencia (Hz)", "Amplitud (dB)", "Probabilidad (%)",
    "Armónicos", "Latencia Red (ms)", "Latencia CNN (ms)", "Alerta"
]

def guardar_en_excel_local(fila: list):
    """Descarga el Excel de GitHub, agrega el registro y lo vuelve a subir."""
    try:
        headers_gh = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        zona_guatemala = pytz.timezone("America/Guatemala")
        fecha_hoy = datetime.now(zona_guatemala).strftime("%Y-%m-%d")
        nombre_excel = f"reporte_{fecha_hoy}.xlsx"
        ruta_github_archivo = f"{GITHUB_PATH_BASE}/{nombre_excel}"
        url_archivo = f"https://github.com{GITHUB_USER}/{GITHUB_REPO}/contents/{ruta_github_archivo}"

        response = req_github.get(url_archivo, headers=headers_gh, timeout=5)
        sha = None

        if response.status_code == 200:
            data      = response.json()
            sha       = data["sha"]  
            contenido = base64.b64decode(data["content"])
            wb        = openpyxl.load_workbook(BytesIO(contenido))
            ws        = wb.active
            print(f"  📥 Excel descargado de GitHub: {nombre_excel}")
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Registros Aedes"
            ws.append(EXCEL_HEADERS)
            print(f"  📄 Creando nuevo Excel para el día: {nombre_excel}")

        ws.append(fila)

        buffer = BytesIO()
        wb.save(buffer)
        contenido_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "message": f"Evento #{fila[0]} registrado en {nombre_excel}",
            "content": contenido_b64
        }
        if sha:
            payload["sha"] = sha  

        put_response = req_github.put(url_archivo, headers=headers_gh, json=payload, timeout=5)

        if put_response.status_code in[200, 201]:  # <--- CORREGIDO
            print(f"  ✔ Excel actualizado con éxito en GitHub: {nombre_excel}")

        else:
            print(f"  ⚠️ Error subiendo Excel a GitHub: {put_response.status_code} — {put_response.text}")
    except Exception as e:
        print(f"  ❌ Error crítico en el proceso de GitHub: {e}")

# ==============================================================================
# --- 3. PROCESAMIENTO ASÍNCRONO EN SEGUNDO PLANO ---
# ==============================================================================
def procesar_datos_pesados(raw_audio, distancia, t_recepcion_esp):
    """Ejecuta la inferencia de IA y el guardado en GitHub sin congelar al ESP32."""
    global contador_evento
    try:
        t_inicio_cnn = time.time()
        
        # Convertir los bytes a arreglo numérico de NumPy para tu modelo
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)
        
        # ----------------------------------------------------------------------
        # [AQUÍ VA TU LÓGICA EXISTENTE DE FILTRADO, PREPROCESAMIENTO Y TU CNN TFLITE]
        # REEMPLAZA ESTAS LÍNEAS SIMULADAS POR LAS VARIABLES DE TU MODELO REAL:
        probabilidad_aedes = 0.88  
        frecuencia_hz = 450.0
        amplitud_db = -42.5
        armonicos = "Ninguno"
        alerta = "Alta"
        # ----------------------------------------------------------------------
        
        t_fin_cnn = time.time()
        latencia_cnn_ms = int((t_fin_cnn - t_inicio_cnn) * 1000)
        latencia_red_ms = int(time.time() * 1000) - t_recepcion_esp

        # Estructurar fecha y hora de Guatemala
        zona_guatemala = pytz.timezone("America/Guatemala")
        ahora = datetime.now(zona_guatemala)
        fecha = ahora.strftime("%Y-%m-%d")
        hora = ahora.strftime("%H:%M:%S")
        
        # Armar fila para el Excel en GitHub
        nueva_fila = [
            contador_evento, fecha, hora, distancia, 
            frecuencia_hz, amplitud_db, probabilidad_aedes * 100, 
            armonicos, latencia_red_ms, latencia_cnn_ms, alerta
        ]
        
        # Guardar en GitHub de forma aislada
        guardar_en_excel_local(nueva_fila)
        
        contador_evento += 1
    except Exception as e:
        print("❌ Error procesando la tarea en segundo plano:")
        traceback.print_exc()

# ==============================================================================
# --- 4. ENDPOINT PRINCIPAL FASTAPI ---
# ==============================================================================
app = FastAPI()

@app.post("/predict")
async def recibir_audio_wifi(request: Request, background_tasks: BackgroundTasks):
    try:
        t_recepcion = int(time.time() * 1000)
        
        # Leer variables rápidas de las cabeceras HTTP del ESP32
        distancia = request.headers.get("X-Distance", "0")
        
        # Captura instantánea de los bytes de audio en memoria RAM
        raw_audio = await request.body()
        
        # Delegar la IA y el Excel a segundo plano (El ESP32 queda libre de inmediato)
        background_tasks.add_task(procesar_datos_pesados, raw_audio, distancia, t_recepcion)
        
        # Respuesta inmediata al ESP32 para evitar desconexiones (Timeout)
        return {"status": "recibido", "hora": datetime.now().strftime("%H:%M:%S"), "latencia": "0ms"}
        
    except Exception as e:
        print("❌ Error en endpoint /predict:")
        traceback.print_exc()
        return {"status": "error", "detalle": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
