"""
Model Architecture Module
Defines CNN-LSTM hybrid model for animal image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

IMG_SIZE = 64
NUM_CLASSES = 10
SEQUENCE_LENGTH = IMG_SIZE  # Each row is a time step
FEATURE_DIM = IMG_SIZE * 3  # Width * Channels (RGB)

def create_cnn_lstm_model(input_shape=(IMG_SIZE, IMG_SIZE * 3), 
                          num_classes=NUM_CLASSES,
                          lstm_units=128,
                          dropout_rate=0.3):
    """
    Create CNN-LSTM hybrid model
    
    Architecture:
    1. Reshape sequence back to image for CNN processing
    2. CNN encoder to extract spatial features
    3. Reshape CNN output to sequences
    4. LSTM layers for temporal processing
    5. Dense classifier
    
    Args:
        input_shape: (sequence_length, features) = (64, 192)
        num_classes: Number of output classes
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Reshape sequence back to image format for CNN
    # (batch, 64, 192) -> (batch, 64, 64, 3)
    x = layers.Reshape((IMG_SIZE, IMG_SIZE, 3))(inputs)
    
    # CNN Encoder - Extract spatial features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Reshape CNN output to sequences for LSTM
    # (batch, 8, 8, 128) -> (batch, 8, 8*128)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # LSTM layers for sequence processing
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.LSTM(lstm_units // 2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Classifier')
    
    return model

def create_pure_lstm_model(input_shape=(IMG_SIZE, IMG_SIZE * 3),
                           num_classes=NUM_CLASSES,
                           lstm_units=128,
                           dropout_rate=0.3):
    """
    Create pure LSTM model (alternative architecture)
    Treats each image row as a sequence step
    
    Args:
        input_shape: (sequence_length, features)
        num_classes: Number of output classes
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.LSTM(lstm_units // 2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Pure_LSTM_Classifier')
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile model with Adam optimizer
    
    Args:
        model: Keras model
        learning_rate: Initial learning rate
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Creating CNN-LSTM model...")
    model = create_cnn_lstm_model()
    model = compile_model(model)
    model.summary()
    
    print("\n" + "="*80 + "\n")
    
    print("Creating Pure LSTM model...")
    model2 = create_pure_lstm_model()
    model2 = compile_model(model2)
    model2.summary()
