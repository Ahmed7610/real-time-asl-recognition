FROM python:3.12-slim

# System libs required by OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ /app/backend/
# Trained model bundle (models, scaler, encoders, asl_*.py, hand_landmarker.task)
COPY ["Real-Time Sign Language Recognition", "/app/models_bundle"]

ENV MODELS_DIR=/app/models_bundle
WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
