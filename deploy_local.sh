#!/bin/bash
set -e

IMAGE_NAME="cv-api"

echo "📦 Building Docker image..."
# Note: No --platform flag needed for local testing on your Mac
docker build -t $IMAGE_NAME .

echo "🚀 Starting container on http://localhost:8000"
echo "   (Press Ctrl+C to stop)"

# --rm: Automatically remove the container when it stops
# -p 8000:8000: Map your Mac's port 8000 to the container's 8000
docker run --rm -p 8000:8000 $IMAGE_NAME
