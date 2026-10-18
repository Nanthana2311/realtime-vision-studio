#!/usr/bin/env bash
# VisionStudio Edge 1-Click Startup Script (macOS / Linux)

echo "🚀 Starting VisionStudio Edge..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Start FastAPI Uvicorn server
echo "✨ Launching FastAPI on http://localhost:8000 ..."
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
