FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY pose_detection_api.py .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "pose_detection_api:app", "--host", "0.0.0.0", "--port", "8000"]
