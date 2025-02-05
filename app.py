from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = FastAPI()

# Enable CORS (Allow all origins - update for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load YOLO Model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH).to('cpu')  # Force CPU mode

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return FileResponse("index.html")  # Serve the HTML file

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded video
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Open video
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Error opening video"})

        accident_detected = False
        detected_frame_path = "detected_frame.jpg"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to a proper format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            results = model(frame_rgb)  # Run YOLO detection

            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    accident_detected = True

                    # Save the detected frame
                    cv2.imwrite(detected_frame_path, frame)
                    break  # Stop processing once an accident is found

        # Cleanup
        cap.release()
        os.remove(temp_file_path)  # Delete video file

        if accident_detected:
            return {
                "label": "Accident Detected",
                "frame_url": "/detected_frame"
            }
        else:
            return {"label": "No Accident"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "message": str(e)})

@app.get("/detected_frame")
async def get_detected_frame():
    """Serve the accident-detected frame."""
    return FileResponse("detected_frame.jpg", media_type="image/jpeg")

# Ensure the correct port is used
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
