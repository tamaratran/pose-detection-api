# Pose Detection API

FastAPI backend for real-time human pose detection using MediaPipe.

## Local Development

```bash
pip install -r requirements.txt
python pose_detection_api.py
```

Visit `http://localhost:8000/docs` for API documentation.

## API Endpoints

### POST `/detect_pose`
Detect pose landmarks in a single image.

**Request:** Form data with `file` (image)

**Response:**
```json
{
  "success": true,
  "landmarks": [[
    {"x": 0.5, "y": 0.2, "z": 0, "visibility": 0.9},
    ...
  ]],
  "count": 33
}
```

### GET `/health`
Health check endpoint.

## Deployment on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select this repository
5. Railway will auto-detect the Dockerfile
6. Deploy!
7. Copy your Railway URL and update the React Native app

## Environment Variables

None required for basic usage. Railway will automatically assign a PORT.

## Model

Uses MediaPipe's pose_landmarker_lite model:
- Downloaded automatically on first run
- ~5MB model file
- Returns 33 landmark points per pose
