FROM python:3.10-slim

WORKDIR /app

# 1. INSTALAR DEPENDENCIAS DEL SISTEMA PARA AUDIO
# Esto es vital para que librosa (sndfile) y numpy puedan procesar los .wav
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. INSTALAR DEPENDENCIAS DE PYTHON
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. COPIAR EL PROYECTO (Incluyendo tu main.py y mi_modelo_aedes.tflite)
COPY . .

# 4. ARRANCAR EL SERVIDOR
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
