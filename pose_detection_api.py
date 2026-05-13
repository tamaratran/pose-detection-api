import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose Landmarker
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = None
pose_landmarker = None

def init_pose_detector():
    global pose_landmarker, model_path

    if pose_landmarker is not None:
        return

    # Try to download the model if not present
    model_path = "pose_landmarker_lite.task"

    if not os.path.exists(model_path):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        print(f"[Init] Downloading model from {url}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("[Init] Model downloaded successfully")
        except Exception as e:
            print(f"[Init] ERROR downloading model: {e}")
            raise
    else:
        print(f"[Init] Model file exists: {model_path}")

    print("[Init] Creating pose landmarker...")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1
    )

    pose_landmarker = PoseLandmarker.create_from_options(options)
    print("[Init] Pose detector initialized successfully")

@app.on_event("startup")
async def startup_event():
    print("[Startup] Starting initialization...")
    init_pose_detector()
    print("[Startup] Initialization complete")

@app.get("/warmup")
async def warmup():
    """Warmup endpoint to ensure model is loaded"""
    if pose_landmarker is None:
        init_pose_detector()
    return {"status": "ready", "model_loaded": pose_landmarker is not None}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/detect_pose")
async def detect_pose(file: UploadFile = File(...)):
    """
    Detect pose landmarks in an image.

    Returns JSON with landmarks array containing:
    - x, y, z: coordinates (0-1 normalized)
    - visibility: confidence score (0-1)
    """
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL image to numpy array
        image_array = np.asarray(image)

        # Create MediaPipe Image using the correct format
        try:
            from mediapipe.framework.formats import image as image_lib
            mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=image_array)
        except ImportError:
            # Fallback: create Image object directly
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

        # Detect pose
        detection_result = pose_landmarker.detect(mp_image)

        # Extract landmarks
        all_landmarks = []

        # In newer MediaPipe, pose landmarks are in pose_landmarks attribute
        if hasattr(detection_result, 'pose_landmarks') and detection_result.pose_landmarks:
            for pose in detection_result.pose_landmarks:
                pose_landmarks = []
                for landmark in pose:
                    pose_landmarks.append({
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "visibility": float(landmark.visibility) if hasattr(landmark, 'visibility') else 0.5
                    })
                if pose_landmarks:
                    all_landmarks.append(pose_landmarks)

        return {
            "success": True,
            "landmarks": all_landmarks,  # Direct array of pose arrays
            "count": len(all_landmarks[0]) if all_landmarks else 0
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/detect_pose_batch")
async def detect_pose_batch(files: list[UploadFile] = File(...)):
    """
    Detect pose in multiple images.
    """
    results = []
    for file in files:
        result = await detect_pose(file)
        results.append(result)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
