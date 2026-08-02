#!/usr/bin/env bash
# Start the Real-Time Sign Language Recognition API.
# Usage: ./run.sh   (from the backend/ folder)
set -e
cd "$(dirname "$0")"

# Point at the trained model bundle (override if you moved it).
export MODELS_DIR="${MODELS_DIR:-../Real-Time Sign Language Recognition}"

exec uvicorn app:app --host 0.0.0.0 --port 8000 --reload
