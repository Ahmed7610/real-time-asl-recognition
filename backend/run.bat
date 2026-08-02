@echo off
REM Start the Real-Time Sign Language Recognition API (Windows).
cd /d "%~dp0"
if "%MODELS_DIR%"=="" set "MODELS_DIR=..\Real-Time Sign Language Recognition"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
