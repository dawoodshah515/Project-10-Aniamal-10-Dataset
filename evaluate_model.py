"""
Model Evaluation Script
Evaluates trained model and generates detailed metrics and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import prepare_data, CLASSES

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        save_path: Path to save the plot
    """
    # Convert one-hot to class indices
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm

def plot_per_class_accuracy(y_true, y_pred, save_path='per_class_accuracy.png'):
    """
    Plot per-class accuracy
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        save_path: Path to save the plot
    """
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    
    # Calculate per-class accuracy
    accuracies = []
    for i in range(len(CLASSES)):
        mask = y_true_idx == i
        if mask.sum() > 0:
            acc = (y_pred_idx[mask] == i).sum() / mask.sum()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASSES, accuracies, color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if acc >= 80:
            bar.set_color('green')
        elif acc >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=70, color='red', linestyle='--', linewidth=2, label='Target (70%)')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Animal Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to {save_path}")
    plt.close()

def evaluate_model(model_path='animal_classifier.h5', dataset_path='Animals-10'):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to saved model
        dataset_path: Path to dataset
    """
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Prepare data
    print("\nPreparing test data...")
    data = prepare_data(dataset_path, test_size=0.2, val_size=0.1)
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"✓ Test set: {len(X_test)} images")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.zeros_like(y_pred_probs)
    y_pred[np.arange(len(y_pred)), y_pred_probs.argmax(1)] = 1
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    test_loss, test_accuracy, test_top3_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Top-3 Accuracy: {test_top3_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    print(classification_report(y_true_idx, y_pred_idx, target_names=CLASSES))
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    cm = plot_confusion_matrix(y_test, y_pred)
    plot_per_class_accuracy(y_test, y_pred)
    
    # Save detailed results
    with open('evaluation_results.txt', 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Overall Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {test_top3_acc*100:.2f}%\n")
        f.write(f"Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true_idx, y_pred_idx, target_names=CLASSES))
    
    print("✓ Evaluation results saved to evaluation_results.txt")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    evaluate_model()
