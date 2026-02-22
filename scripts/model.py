import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import os

def create_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model for Emotion Recognition.
    
    Args:
        input_shape (tuple): Shape of the input data (height, width, channels).
                             For RAVDESS Mel-spectrograms, it's typically (128, X, 1).
        num_classes (int): Number of emotion categories to predict.
        
    Returns:
        model (tf.keras.Model): Compiled Keras CNN model.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flattening to feed into Dense layers
        Flatten(),
        
        # Fully Connected Block
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                   
    return model

if __name__ == "__main__":
    # Test model creation with the processed data shape
    # Example input shape based on features: (128, 130, 1)
    # RAVDESS has 8 emotion classes
    input_shape = (128, 130, 1)
    num_classes = 8
    
    model = create_model(input_shape, num_classes)
    model.summary()
