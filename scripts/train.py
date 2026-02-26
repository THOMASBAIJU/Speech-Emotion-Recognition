import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Import the create_model function
from model import create_model

def augment_data(X, y):
    """Augment data with Gaussian noise."""
    noise_factor = 0.005 # mild noise
    X_augmented = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    # Clip values
    X_augmented = np.clip(X_augmented, np.min(X), np.max(X))
    
    # Append augmented
    X_combined = np.concatenate((X, X_augmented), axis=0)
    y_combined = np.concatenate((y, y), axis=0)
    
    return X_combined, y_combined

def main():
    features_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'X.npy')
    labels_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'y.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Processed data not found. Please run preprocess_data.py first.")
        return
        
    print("Loading extracted features...")
    X = np.load(features_path)
    y = np.load(labels_path)
    
    # Label encoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    print(f"Classes found: {encoder.classes_}")
    
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 2. Data Augmentation
    print("Augmenting data...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"Training set after augmentation: {X_train.shape[0]} samples")
    
    # 3. Model Architecture
    input_shape = X_train.shape[1:]
    model = create_model(input_shape, num_classes)
    
    # 4. Training
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # 5. Evaluation
    print("Evaluating model performance on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.save(os.path.join(model_dir, 'emotion_model.h5'))
    print("Model saved to models/emotion_model.h5")

if __name__ == '__main__':
    main()
