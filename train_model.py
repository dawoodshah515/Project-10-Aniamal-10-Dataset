"""
Training Script for Animals-10 RNN Classifier
Trains CNN-LSTM hybrid model with early stopping and model checkpointing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_preprocessing import prepare_data, CLASSES
from model_architecture import create_cnn_lstm_model, compile_model

# Configuration
DATASET_PATH = "Animals-10"
MODEL_SAVE_PATH = "saved_model"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# For faster testing, set max_images_per_class (e.g., 500)
# For full training, set to None
MAX_IMAGES_PER_CLASS = None  # Set to None for full dataset

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()

def train_model():
    """
    Main training function
    """
    print("="*80)
    print("ANIMALS-10 RNN CLASSIFIER TRAINING")
    print("="*80)
    
    # Prepare data
    print("\n[1/5] Preparing dataset...")
    data = prepare_data(
        DATASET_PATH, 
        test_size=0.2, 
        val_size=0.1,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\nDataset prepared successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("\n[2/5] Building CNN-LSTM model...")
    model = create_cnn_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=len(CLASSES),
        lstm_units=128,
        dropout_rate=0.3
    )
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print("\nModel architecture:")
    model.summary()
    
    # Setup callbacks
    print("\n[3/5] Setting up training callbacks...")
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    # Model checkpoint
    checkpoint_path = 'best_model.h5'
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [early_stop, checkpoint, reduce_lr]
    
    # Train model
    print("\n[4/5] Training model...")
    print(f"Target: ≥70% validation accuracy")
    print(f"Max epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-"*80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_accuracy, test_top3_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Final Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Top-3 Accuracy: {test_top3_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Check if target accuracy achieved
    if test_accuracy >= 0.70:
        print("\n✓ Target accuracy (≥70%) ACHIEVED!")
    else:
        print(f"\n⚠ Target accuracy not reached. Consider training longer or tuning hyperparameters.")
    
    # Save final model
    print("\nSaving model...")
    
    # Save in SavedModel format
    model.save(MODEL_SAVE_PATH)
    print(f"✓ Model saved to {MODEL_SAVE_PATH}/")
    
    # Save as H5 format (for easier loading)
    model.save('animal_classifier.h5')
    print(f"✓ Model saved to animal_classifier.h5")
    
    # Plot training history
    plot_training_history(history)
    
    # Save training info
    with open('training_info.txt', 'w') as f:
        f.write("Animals-10 RNN Classifier Training Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Animals-10\n")
        f.write(f"Classes: {', '.join(CLASSES)}\n\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Model: CNN-LSTM Hybrid\n")
        f.write(f"Epochs trained: {len(history.history['accuracy'])}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Initial learning rate: {LEARNING_RATE}\n\n")
        f.write("Results:\n")
        f.write(f"  Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%\n")
        f.write(f"  Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"  Test Top-3 Accuracy: {test_top3_acc*100:.2f}%\n")
        f.write(f"  Test Loss: {test_loss:.4f}\n")
    
    print("✓ Training info saved to training_info.txt")
    
    print("\n" + "="*80)
    print("All files saved successfully!")
    print("="*80)
    
    return model, history

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model, history = train_model()
