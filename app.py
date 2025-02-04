from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = FastAPI()

# Enabling CORS to allow all origins (replace '*' with specific domains if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins; change if needed for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH).to('cpu')  # Force CPU mode for debugging

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return FileResponse("index.html")  # Serving the HTML file

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded video to a temporary file
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Open the video
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            return {"error": "Error opening video"}

        accident_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on each frame
            results = model(frame)

            # Check for detection results
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    accident_detected = True
                    break  # If accident is detected in any frame, stop further processing

        # Close video file after processing
        cap.release()
        os.remove(temp_file_path)  # Clean up the temporary file

        return {"label": "Accident Detected" if accident_detected else "No Accident"}

    except Exception as e:
        return {"error": "Internal Server Error", "message": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
        return {"error": "Internal Server Error", "message": str(e)}
