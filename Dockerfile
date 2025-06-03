# Use official Python image with necessary dependencies
FROM python:3.10-slim-bullseye

# Install system dependencies (including git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AI dependencies (now with git available)
RUN pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/CLIP
RUN pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip

# Download AI models
RUN wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
RUN wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Start application
CMD ["flask", "run"]
