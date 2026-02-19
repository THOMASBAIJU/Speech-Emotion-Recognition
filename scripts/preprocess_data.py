import os
import glob
import librosa
import numpy as np
import pandas as pd
import pickle

# Emotions in the RAVDESS dataset
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Intensity (01 = normal, 02 = strong).
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd=Male, Even=Female).

def extract_features(file_name):
    """
    Extracts features (mfcc, chroma, mel) from a sound file.
    """
    # Load audio file with a fixed duration (e.g., 3 seconds) usually around 2.5-3s in RAVDESS
    target_time = 3 # seconds
    try:
        y, sr = librosa.load(file_name, duration=target_time, offset=0.5)
        
        # Pad if shorter than target_time
        target_length = int(target_time * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            y = y[:target_length]
            
        # Mel Spectrogram
        # n_mels=128 (Height of image)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Add channel dimension (like grayscale image)
        # Shape: (128, Time, 1)
        mel_spect_db = mel_spect_db[..., np.newaxis]
        
        return mel_spect_db

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def load_ravdess_data(data_path):
    features = []
    labels = []
    
    # Iterate through all files in the RAVDESS directory structure
    # RAVDESS structure is usually: Actor_01/03-01-01-01-01-01-01.wav
    for file in glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True):
        file_name = os.path.basename(file)
        
        # Extract emotion from filename
        # 03-01-06-01-02-01-12.wav
        parts = file_name.split("-")
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTIONS.get(emotion_code)
            
            if emotion:
                feature = extract_features(file)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
            else:
                print(f"Skipping {file_name}: Unknown emotion code {emotion_code}")
        else:
            print(f"Skipping {file_name}: Invalid filename format")
    
    return features, labels

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'RAVDESS')
    
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return

    print("Extracting features from RAVDESS dataset...")
    features, labels = load_ravdess_data(data_path)
    print(f"Found {len(features)} valid audio files.")
    
    if len(features) == 0:
        print("No features extracted. Please check the data directory and filenames.")
        return

    X = np.array(features)
    y = np.array(labels)
    
    print(f"Features extracted: {X.shape[0]} samples")
    try:
        print(f"Feature vector size: {X.shape[1]}")
    except IndexError:
        print("Feature vector size: Undefined (Empty array)")

    # Save processed data
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'processed_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    main()
