FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Using CPU-only versions to keep the image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu torch==2.4.0+cpu torchvision==0.19.0+cpu

# Copy project files
COPY api.py task1.py load_oxford_flowers102.py /app/
# Copy model weights (make sure these exist in your local 'saved' folder)
COPY saved/CNN_fine.weights.h5 /app/saved/CNN_fine.weights.h5
COPY saved/CNN_coarse.weights.h5 /app/saved/CNN_coarse.weights.h5

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
