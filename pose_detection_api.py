import os
import io
import base64
import urllib.request
from fastapi import FastAPI, File, UploadFile, Request
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

def extract_body_outline(landmarks):
    """Extract body outline/silhouette from pose landmarks.

    Returns a sequence of landmark indices that trace the body outline.
    """
    # Body outline connection sequence: traces around the body
    outline_indices = [
        # Head
        0, 1, 2, 3, 4, 5, 6, 7,
        # Right side (arm to hip)
        10, 12, 14, 16, 18, 20, 22, 26, 28, 30, 32,
        # Bottom right leg
        32, 31, 29, 27,
        # Left side (hip to arm)
        23, 25, 24, 11, 13, 15, 17, 19, 21,
        # Back to head
        9, 10, 0
    ]

    # Filter to only visible landmarks and return points
    outline = []
    for idx in outline_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            if lm.get('visibility', 0.5) > 0.2:  # Only include visible points
                outline.append({
                    'x': lm['x'],
                    'y': lm['y'],
                    'z': lm['z'],
                    'visibility': lm['visibility']
                })

    return outline

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
        all_outlines = []

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
                    # Extract body outline from landmarks
                    outline = extract_body_outline(pose_landmarks)
                    all_outlines.append(outline)

        return {
            "success": True,
            "landmarks": all_landmarks,  # Direct array of pose arrays
            "outline": all_outlines,  # Body silhouette outline
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


def detect_from_image_bytes(image_data: bytes):
    """Run pose detection on raw image bytes. Returns MediaPipe-indexed landmarks dict or None."""
    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.asarray(image)

    try:
        from mediapipe.framework.formats import image as image_lib
        mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=image_array)
    except ImportError:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    detection_result = pose_landmarker.detect(mp_image)

    if not (hasattr(detection_result, 'pose_landmarks') and detection_result.pose_landmarks):
        return None

    pose = detection_result.pose_landmarks[0]
    landmarks = {}
    for i, lm in enumerate(pose):
        landmarks[i] = {
            "x": round(float(lm.x), 3),
            "y": round(float(lm.y), 3),
        }
    return landmarks


@app.post("/detectPoseLandmarksBatch")
async def detect_pose_landmarks_batch(request: Request):
    """
    Firebase-compatible batch endpoint.
    Accepts: { images: [{ key: string, imageUrl: string }] }
    Returns: { results: { [key]: { [landmarkIndex]: { x, y } } | null } }
    """
    body = await request.json()
    images = body.get("images", [])
    results = {}

    for item in images:
        key = item.get("key")
        image_url = item.get("imageUrl")
        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "PoseAPI/1.0"})
            with urllib.request.urlopen(req) as resp:
                image_data = resp.read()
            landmarks = detect_from_image_bytes(image_data)
            results[key] = landmarks
        except Exception as e:
            print(f"Error detecting pose for {key}: {e}")
            results[key] = None

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
