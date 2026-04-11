import os
import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import model architecture and configurations from task1
from task1 import CNN

# Import class mappings from the dataloader
from load_oxford_flowers102 import flowers102_class_names, flowers102_group_names

app = FastAPI(
    title="CNN Image Classifier API",
    description="Exposes the task1.py CNN PyTorch model to classify flower images."
)

# Add CORS Middleware to allow React app to talk to the API 
# (don't need junyishen.com, because production React app in same domain/port, and use relative paths)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173", # For npm run preview
        "http://localhost:5173", # For npm run dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/cv/health")
def health_check():
    return {"status": "ok"}

# --- 1. Load the Model Configuration ---
# Only use fine-grained, because every image has a fine-grained label, but not all images have coarse-grained labels
fine_grained = True 

if fine_grained:
    n_classes = len(flowers102_class_names)
    class_names = flowers102_class_names
    weights_path = os.path.join("saved", "CNN_fine.weights.h5")
else:
    n_classes = len(flowers102_group_names)
    class_names = list(flowers102_group_names.keys())
    weights_path = os.path.join("saved", "CNN_coarse.weights.h5")

# print ("class_names: ", class_names)

# Hardcoded for the model trained in task1.py
reg_dropout_rate = 0 
reg_batch_norm = True 

# Use GPU/MPS if available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- 2. Initialize and Load the Model ---
classifier_model = CNN(
    in_channels=3, 
    n_classes=n_classes, 
    reg_dropout_rate=reg_dropout_rate, 
    reg_batch_norm=reg_batch_norm
)
classifier_model.to(device)

# Load weights if the file exists
if os.path.isfile(weights_path):
    print(f"Loading weights from {weights_path}")
    classifier_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
else:
    print(f"Warning: weights file not found at {weights_path}.")

# It is critical to set the model to evaluation mode for inference
classifier_model.eval()

# --- 3. Image Preprocessing ---
# Based on the test transform used in load_oxford_flowers102.py
imsize = 96
transform = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()
])

# --- 4. Define the API Endpoint ---
@app.post("/api/cv/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be an image.")
    
    try:
        # Read image to memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image and add batch dimension (B, C, H, W)
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = classifier_model(tensor)
            # The model outputs raw logits, argmax gives the highest prob class ID
            prediction_index = torch.argmax(output, dim=1).item()
            
        predicted_class_name = class_names[prediction_index]
        
        return {
            "filename": file.filename,
            "prediction_index": prediction_index,
            "prediction_class": predicted_class_name,
            "model_type": "fine-grained" if fine_grained else "coarse"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Optional wrapper for running via python directly: `python api.py`
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
