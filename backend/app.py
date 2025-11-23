"""
FastAPI Backend for Animals-10 Classifier
Provides /predict endpoint for image classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

# Configuration
MODEL_PATH = "../animal_classifier.h5"  # Adjust path as needed
IMG_SIZE = 64
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Initialize FastAPI app
app = FastAPI(
    title="Animals-10 Classifier API",
    description="RNN-based deep learning model for animal image classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return model

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model inference
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Preprocessed image ready for model input
    """
    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    img_array = np.array(image, dtype='float32')
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Convert to sequence format (same as training)
    # Shape: (64, 64, 3) -> (64, 192)
    img_seq = img_array.reshape(IMG_SIZE, IMG_SIZE * 3)
    
    # Add batch dimension
    img_seq = np.expand_dims(img_seq, axis=0)
    
    return img_seq

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Animals-10 Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Health check",
            "/classes": "GET - List available classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    return {
        "classes": CLASSES,
        "num_classes": len(CLASSES)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict animal class from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Load model if not already loaded
        if model is None:
            load_model()
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        top_class = CLASSES[top_idx]
        top_confidence = float(predictions[0][top_idx])
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [
            {
                "class": CLASSES[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top3_indices
        ]
        
        # Get all class probabilities
        all_predictions = [
            {
                "class": CLASSES[i],
                "confidence": float(predictions[0][i])
            }
            for i in range(len(CLASSES))
        ]
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class": top_class,
                "confidence": top_confidence
            },
            "top3": top3_predictions,
            "all_predictions": all_predictions
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("ANIMALS-10 CLASSIFIER API")
    print("="*80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("\nStarting server...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
