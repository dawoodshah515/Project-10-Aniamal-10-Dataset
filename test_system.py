"""
Quick Test Script
Tests the data preprocessing and model architecture without full training
"""

import numpy as np
from data_preprocessing import prepare_data, CLASSES
from model_architecture import create_cnn_lstm_model, compile_model

print("="*80)
print("ANIMALS-10 SYSTEM TEST")
print("="*80)

# Test 1: Data Preprocessing
print("\n[Test 1] Data Preprocessing")
print("-"*80)
try:
    print("Loading small sample of data (100 images per class)...")
    data = prepare_data("Animals-10", test_size=0.2, val_size=0.1, max_images_per_class=100)
    
    print(f"✓ Data loaded successfully!")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Input shape: {data['X_train'].shape}")
    print(f"  Classes: {len(CLASSES)}")
except Exception as e:
    print(f"✗ Data preprocessing failed: {e}")
    exit(1)

# Test 2: Model Architecture
print("\n[Test 2] Model Architecture")
print("-"*80)
try:
    print("Creating CNN-LSTM model...")
    model = create_cnn_lstm_model(
        input_shape=(data['X_train'].shape[1], data['X_train'].shape[2]),
        num_classes=len(CLASSES)
    )
    model = compile_model(model)
    
    print(f"✓ Model created successfully!")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    exit(1)

# Test 3: Model Inference
print("\n[Test 3] Model Inference")
print("-"*80)
try:
    print("Testing model prediction with random data...")
    sample_input = data['X_train'][:1]
    prediction = model.predict(sample_input, verbose=0)
    
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    print(f"✓ Inference successful!")
    print(f"  Predicted class: {CLASSES[predicted_class]}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Output shape: {prediction.shape}")
except Exception as e:
    print(f"✗ Inference failed: {e}")
    exit(1)

# Summary
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nThe system is ready for training.")
print("\nNext steps:")
print("1. Run 'python train_model.py' to train the model")
print("2. Run 'python backend/app.py' to start the API")
print("3. Open 'frontend/index.html' in your browser")
print("="*80)
