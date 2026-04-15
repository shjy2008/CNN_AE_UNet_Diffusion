import time 
start_time = time.time()

import os
import io
import logging
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

load_time = time.time()
print(f">>> Initial import lightweight part took {load_time - start_time:.3f}s")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
classifier_model = None
ae_encoder = None
ae_decoder = None
denoiser_model = None
device = None
class_names = None
fine_grained = True
transform = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_classifier():
    global classifier_model, class_names, fine_grained, device
    fine_grained = True
    # Lazy-import dataset mappings and model class to avoid heavy top-level imports
    from load_oxford_flowers102 import flowers102_class_names, flowers102_group_names
    from task1 import CNN

    if fine_grained:
        n_classes = len(flowers102_class_names)
        class_names = flowers102_class_names
        weights_path = os.path.join("saved", "CNN_fine.weights.h5")
    else:
        n_classes = len(flowers102_group_names)
        class_names = list(flowers102_group_names.keys())
        weights_path = os.path.join("saved", "CNN_coarse.weights.h5")

    # Initialize the Model
    classifier_model = CNN(
        in_channels=3, 
        n_classes=n_classes, 
        reg_dropout_rate=0, 
        reg_batch_norm=True
    )
    classifier_model.to(device)

    # Load weights using memory mapping for significantly faster initialization
    if os.path.isfile(weights_path):
        print(f"Loading weights from {weights_path}")
        classifier_model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True, mmap=True)
        )
    else:
        print(f"Warning: weights file not found at {weights_path}.")

    classifier_model.eval()


def load_generators():
    global ae_encoder, ae_decoder, denoiser_model, device
    # Lazy-import generator model classes
    from task2a import AE_Encoder, AE_Decoder
    from task2b import UNetDenoiser

    ae_encoder = AE_Encoder(in_channels=3)
    ae_decoder = AE_Decoder(out_channels=3)
    denoiser_model = UNetDenoiser()

    ae_encoder.to(device)
    ae_decoder.to(device)
    denoiser_model.to(device)

    ae_encoder_weights = os.path.join("saved", "AE_encoder.weights.h5")
    ae_decoder_weights = os.path.join("saved", "AE_decoder.weights.h5")
    denoiser_weights = os.path.join("saved", "Denoiser.weights.h5")

    # Load local generation weights using memory mapping
    if os.path.isfile(ae_encoder_weights):
        print(f"Loading weights from {ae_encoder_weights}")
        ae_encoder.load_state_dict(torch.load(ae_encoder_weights, map_location=device, weights_only=True, mmap=True))
    if os.path.isfile(ae_decoder_weights):
        print(f"Loading weights from {ae_decoder_weights}")
        ae_decoder.load_state_dict(torch.load(ae_decoder_weights, map_location=device, weights_only=True, mmap=True))
    if os.path.isfile(denoiser_weights):
        print(f"Loading weights from {denoiser_weights}")
        denoiser_model.load_state_dict(torch.load(denoiser_weights, map_location=device, weights_only=True, mmap=True))

    ae_encoder.eval()
    ae_decoder.eval()
    denoiser_model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, transform
    lifespan_start = time.time()
    
    device = get_device()
    
    # Image Preprocessing
    imsize = 96
    # Lazy-import torchvision transforms to avoid import cost at module import
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()
    ])

    # Run heavy PyTorch model initialization concurrently in separate threads
    await asyncio.gather(
        asyncio.to_thread(load_classifier),
        asyncio.to_thread(load_generators)
    )
    
    end_time = time.time()
    logger.info(f"===== Lifespan (Concurrent Model Loading) took {end_time - lifespan_start:.2f} seconds =====")
    logger.info(f"===== TOTAL Cold Start (from import to ready) took {end_time - start_time:.2f} seconds =====")
    yield


app = FastAPI(
    title="CNN Image Classifier API",
    description="Exposes the task1.py CNN PyTorch model to classify flower images.",
    lifespan=lifespan
)

# Add CORS Middleware to allow React app to talk to the API 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173", # For npm run preview (vite)
        "http://localhost:5173", # For npm run dev (vite)
        "http://localhost:3000", # For npm run dev/start (Next.js)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/cv/health")
def health_check():
    return {"status": "ok"}


# --- 4. Define the API Endpoint ---
@app.post("/api/cv/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")
    
    try:
        # Read image to memory
        contents = await file.read()
        from PIL import Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image and add batch dimension (B, C, H, W)
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = classifier_model(tensor)
            # The model outputs raw logits, apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction_index_tensor = torch.max(probabilities, dim=1)
            prediction_index = prediction_index_tensor.item()
            confidence = confidence.item()
            
        predicted_class_name = class_names[prediction_index]
        
        return {
            "filename": file.filename,
            "prediction_index": prediction_index,
            "prediction_class": predicted_class_name,
            "confidence": confidence,
            "model_type": "fine-grained" if fine_grained else "coarse"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/api/cv/generate")
async def generate_image(format: str = "gif", batch_size: int = 1):
    try:
        denoise_steps = 10
        imsize = 96
        
        # 1. Update random noise to support batch dimension
        base_image = torch.ones(batch_size, 3, imsize, imsize, device=device) * 0.5
        noisy_image = base_image
        for i in range(denoise_steps):
            std = 0.05 * (i + 1)
            noise = torch.randn(noisy_image.size(), device=device) * std
            noisy_image = noisy_image + noise
            noisy_image = torch.clamp(noisy_image, 0, 1)
            
        random_noise = noisy_image
        
        with torch.no_grad():
            current_latent = ae_encoder(random_noise)
            denoised_images = [random_noise] # List of tensors of shape (B, 3, H, W)
            
            for _ in range(denoise_steps):
                current_latent = denoiser_model(current_latent)
                decoded_image = ae_decoder(current_latent)
                denoised_images.append(torch.clamp(decoded_image, 0, 1))
        
        if format == "gif":
            import torchvision
            pil_images = []
            
            # Ensure we are going from Noise -> Clean
            # If your loop already does this, you don't need the [::-1]
            images_to_process = denoised_images 
            # images_to_process = denoised_images[::-1] # Uncomment this if it still ends in noise
            
            nrow = int(batch_size ** 0.5) if batch_size > 1 else 1
            for step_batch in images_to_process:
                # Normalize if necessary (ensure values are 0-1)
                grid = torchvision.utils.make_grid(step_batch, nrow=nrow, padding=2, normalize=True)
                pil_images.append(transforms.ToPILImage()(grid.cpu()))
                
            buf = io.BytesIO()
            # 500ms per step, then linger on the final clean image
            durations = [500] * (len(pil_images) - 1) + [65535] 
            
            pil_images[0].save(
                buf, format="GIF", save_all=True, append_images=pil_images[1:], 
                duration=durations, 
                loop=1  # Set to 1 to play once, though frontend will handle the 'freeze' too
            )
            buf.seek(0)
            media_type = "image/gif"

        else:
            import torchvision
            if batch_size == 1:
                # Show process of 1 image
                all_images_tensor = torch.stack(denoised_images)
                grid = torchvision.utils.make_grid(all_images_tensor, nrow=denoise_steps + 1, padding=2)
            else:
                # Show only the final step for all generated images in grid format
                final_batch = denoised_images[-1]
                nrow = int(batch_size ** 0.5)
                grid = torchvision.utils.make_grid(final_batch, nrow=nrow, padding=2)
                
            grid_clamped = torch.clamp(grid, 0, 1)
            pil_image = transforms.ToPILImage()(grid_clamped.cpu())
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG")
            media_type = "image/jpeg"
        
        return Response(content=buf.getvalue(), media_type=media_type, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Optional wrapper for running via python directly: `python api.py`
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
