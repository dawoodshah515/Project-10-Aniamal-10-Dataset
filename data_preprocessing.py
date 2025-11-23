"""
Data Preprocessing Module for Animals-10 Dataset
Handles image loading, preprocessing, augmentation, and sequence conversion for RNN input
"""

import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Dataset configuration
IMG_SIZE = 64
BATCH_SIZE = 32
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']
NUM_CLASSES = len(CLASSES)

def load_dataset(dataset_path, max_images_per_class=None):
    """
    Load images from the Animals-10 dataset
    
    Args:
        dataset_path: Path to Animals-10 directory
        max_images_per_class: Optional limit for faster testing
    
    Returns:
        X: Array of images
        y: Array of labels
    """
    print("Loading dataset...")
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_name} not found!")
            continue
            
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    X = np.array(images, dtype='float32')
    y = np.array(labels)
    
    print(f"Loaded {len(X)} images across {NUM_CLASSES} classes")
    return X, y

def preprocess_images(X):
    """
    Normalize images to [0, 1] range
    
    Args:
        X: Array of images
    
    Returns:
        Normalized images
    """
    return X / 255.0

def images_to_sequences(X):
    """
    Convert images to sequences for RNN processing
    Each row of the image becomes a time step
    
    Args:
        X: Array of images (N, H, W, C)
    
    Returns:
        Sequences (N, H, W*C) - each row is a time step
    """
    N, H, W, C = X.shape
    # Reshape so each row becomes a sequence step
    X_seq = X.reshape(N, H, W * C)
    return X_seq

def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Create data augmentation generators for training
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        train_gen, val_gen: Data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
    
    return train_gen, val_gen

def prepare_data(dataset_path, test_size=0.2, val_size=0.1, max_images_per_class=None):
    """
    Complete data preparation pipeline
    
    Args:
        dataset_path: Path to Animals-10 directory
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        max_images_per_class: Optional limit for faster testing
    
    Returns:
        Dictionary with train/val/test splits
    """
    # Load dataset
    X, y = load_dataset(dataset_path, max_images_per_class)
    
    # Normalize images
    X = preprocess_images(X)
    
    # Convert labels to categorical
    y_cat = to_categorical(y, NUM_CLASSES)
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"\nData split:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    
    # Convert to sequences for RNN
    X_train_seq = images_to_sequences(X_train)
    X_val_seq = images_to_sequences(X_val)
    X_test_seq = images_to_sequences(X_test)
    
    return {
        'X_train': X_train_seq,
        'y_train': y_train,
        'X_val': X_val_seq,
        'y_val': y_val,
        'X_test': X_test_seq,
        'y_test': y_test,
        'X_train_img': X_train,  # Keep original images for augmentation
        'X_val_img': X_val
    }

if __name__ == "__main__":
    # Test the preprocessing pipeline
    dataset_path = "Animals-10"
    data = prepare_data(dataset_path, max_images_per_class=100)
    
    print(f"\nSequence shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
