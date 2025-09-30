FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Gerekli klasörleri oluştur
RUN mkdir -p data/ai_models_v2 && \
    mkdir -p templates && \
    mkdir -p static

# Port ayarı (Railway otomatik PORT environment variable kullanır)
ENV PORT=8000

# Uygulamayı başlat
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
