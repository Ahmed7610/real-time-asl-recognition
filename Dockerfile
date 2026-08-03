FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELS_DIR="/app/Real-Time Sign Language Recognition" \
    FRONTEND_DIR="/app/frontend"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY ["Real-Time Sign Language Recognition", "/app/Real-Time Sign Language Recognition"]

EXPOSE 10000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
