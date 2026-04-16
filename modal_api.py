import modal
import os

# 1. Define the image with all dependencies
# We use a standard debian slim image and install the necessary Python packages.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "python-multipart",
        "numpy",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
    )
)

# Optimization: Only add necessary files to the container image
base_path = os.path.dirname(os.path.abspath(__file__))

# Add local model weight files individually (only the 5 .h5 files)
h5_files = [
    "CNN_fine.weights.h5",
    "CNN_coarse.weights.h5",
    "AE_encoder.weights.h5",
    "AE_decoder.weights.h5",
    "Denoiser.weights.h5"
]
for f in h5_files:
    image = image.add_local_file(
        os.path.join(base_path, "saved", f), 
        remote_path=f"/root/app/saved/{f}"
    )

# Add the necessary Python code files individually
py_files = ["api.py", "task1.py", "task2a.py", "task2b.py", "load_oxford_flowers102.py"]
for f in py_files:
    image = image.add_local_file(os.path.join(base_path, f), remote_path=f"/root/app/{f}")

# 2. Define the App
app = modal.App("flower-vision-api")

@app.function(
    image=image,
    # gpu="any", # You can specify "T4", "A10G", etc. if you want to use GPU for inference
    scaledown_window=300, # Keeps the container warm for 5 minutes after a request
)
@modal.asgi_app(requires_proxy_auth=True)
def web_app():
    """
    This function wraps your existing FastAPI 'app' from api.py and exposes it as a Modal web endpoint.
    """
    import time
    t0 = time.time()
    print(">>> Modal container starting initialization...")

    import sys
    import os
    # Add the mounted directory to the python path
    sys.path.append("/root/app")
    # Set the working directory to where the app and 'saved' folder are
    os.chdir("/root/app")
    
    t_env = time.time()
    print(f">>> Environment setup took {t_env - t0:.3f}s")

    # Import your local fastapi app
    # This triggers imports and model loading in api.py
    from api import app as fastapi_app
    
    t_import = time.time()
    print(f">>> Importing api.py and loading models took {t_import - t_env:.3f}s")
    print(f">>> Total Python initialization took {t_import - t0:.3f}s")
    
    return fastapi_app
