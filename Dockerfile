FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Using CPU-only versions to keep the image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu torch==2.4.0+cpu torchvision==0.19.0+cpu

# Copy model weights (make sure these exist in your local 'saved' folder)
COPY saved/CNN_fine.weights.h5 /app/saved/CNN_fine.weights.h5
COPY saved/CNN_coarse.weights.h5 /app/saved/CNN_coarse.weights.h5
COPY saved/AE_encoder.weights.h5 /app/saved/AE_encoder.weights.h5
COPY saved/AE_decoder.weights.h5 /app/saved/AE_decoder.weights.h5
COPY saved/Denoiser.weights.h5 /app/saved/Denoiser.weights.h5

# Copy project files
COPY api.py task1.py task2a.py task2b.py load_oxford_flowers102.py /app/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
