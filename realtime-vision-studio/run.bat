@echo off
REM VisionStudio Edge 1-Click Startup Script (Windows)

echo ==============================================
echo 🚀 Launching VisionStudio Edge Platform
echo ==============================================

IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Launching FastAPI server...
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

pause
