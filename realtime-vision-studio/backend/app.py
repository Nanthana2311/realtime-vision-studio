import asyncio
import base64
import csv
import io
import json
import os
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from vision_engine import VisionEngine

engine: VisionEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🚀 [VisionEngine] Booting real-time Computer Vision engine...")
    engine = VisionEngine()
    print("✅ [VisionEngine] Model weights and cascades loaded into memory.")
    yield
    print("🛑 [VisionEngine] Gracefully shutting down vision pipeline.")

app = FastAPI(
    title="Real-Time Edge Vision Studio API",
    description="High-performance WebSocket & REST API for live computer vision inference",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return "<h1>Vision Studio Backend Online. Frontend not built yet.</h1>"

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "Edge Vision Studio",
        "engine_ready": engine is not None
    }

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def decode_base64_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(image_bytes)

@app.post("/api/detect")
async def detect_image(
    file: UploadFile = File(...),
    mode: str = Form("face"),
    confidence: float = Form(0.5)
):
    contents = await file.read()
    frame = decode_image_bytes(contents)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    detections, inference_time_ms, meta = await asyncio.to_thread(
        engine.process_frame, frame, mode, confidence
    )

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": inference_time_ms,
        "meta": meta
    }

@app.post("/api/reset-counter")
async def reset_counter():
    if engine:
        engine.tracker.crossings_count = 0
        engine.tracker.objects.clear()
        engine.tracker.crossed_ids.clear()
    return {"status": "counter_reset", "crossings": 0}

@app.get("/api/export-csv")
async def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Class Label", "Confidence", "Bounding Box [x,y,w,h]", "Total Crossings"])

    for item in list(engine.detection_history):
        writer.writerow([
            item["timestamp"],
            item["label"],
            item["confidence"],
            item["bbox"],
            item["crossings"]
        ])

    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vision_detections.csv"}
    )

@app.websocket("/ws/vision")
async def websocket_vision_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)

            frame_data = payload.get("frame")
            mode = payload.get("mode", "face")
            conf_thresh = float(payload.get("confidence", 0.5))
            tripwire_pos = float(payload.get("tripwire_pos", 0.5))
            client_ts = payload.get("timestamp", time.time() * 1000)

            if not frame_data:
                continue

            frame = decode_base64_image(frame_data)
            if frame is None:
                continue

            # Run inference in threadpool to keep async event loop non-blocking
            detections, inference_time_ms, meta = await asyncio.to_thread(
                engine.process_frame, frame, mode, conf_thresh, tripwire_pos
            )

            response = {
                "detections": detections,
                "count": len(detections),
                "inference_time_ms": inference_time_ms,
                "client_timestamp": client_ts,
                "server_timestamp": time.time() * 1000,
                "meta": meta
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
