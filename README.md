# Speech Emotion Recognition

## Description
This project aims to create a system that detects emotions (e.g., happiness, sadness, anger) from speech recordings by analyzing vocal features. We utilize Deep Learning techniques, specifically Convolutional Neural Networks (CNNs), to analyze Mel-spectrograms extracted from audio files.

## Dataset
We are using the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent.

## Technologies
- **Python**
- **TensorFlow / Keras**: For building and training the neural network.
- **Librosa**: For audio analysis and feature extraction.
- **Matplotlib / Seaborn**: For data visualization.
- **Pandas / NumPy**: For data manipulation.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (see specific instructions in `scripts/` or `data/` directory).

## Usage
1. First, preprocess the audio data to extract Mel-spectrograms:
   ```bash
   python scripts/preprocess_data.py
   ```
2. Next, train the CNN model:
   ```bash
   python scripts/train.py
   ```
The trained model will be saved as an `.h5` artifact in the `models/` directory.

## Solution Approach
This project operates by extracting Mel-spectrograms from raw audio to represent the frequency and time characteristics of speech. These representations visually contain features that a Convolutional Neural Network (CNN) can interpret. With our neural network, built via Keras/TensorFlow, we treat the audio signal visually and distinguish varying underlying emotions like Happiness, Sadness, Angry, Disgust, etc. Background Gaussian noise is also synthetically added into the training split to improve real-world stability.
