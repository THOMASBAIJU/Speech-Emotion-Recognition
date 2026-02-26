import gradio as gr
import numpy as np
import librosa
import os
import tensorflow as tf

# Emotion mappings
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_single_feature(audio_file_path):
    """Extract Mel-spectrogram features from a single audio file."""
    target_time = 3 # seconds
    try:
        y, sr = librosa.load(audio_file_path, duration=target_time, offset=0.5)
        
        target_length = int(target_time * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            y = y[:target_length]
            
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        mel_spect_db = mel_spect_db[..., np.newaxis]
        
        return np.expand_dims(mel_spect_db, axis=0)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_emotion(audio_path):
    if audio_path is None:
        return "Please upload an audio file."
        
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model.h5')
    if not os.path.exists(model_path):
        return f"Model not found at {model_path}. Please train the model first!"

    features = extract_single_feature(audio_path)
    if features is None:
        return "Failed to extract features from the audio."

    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(features)
    
    # Return prob dict
    results = {}
    for idx, label in enumerate(EMOTION_LABELS):
        results[label] = float(prediction[0][idx])
        
    return results

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload Audio File (.wav) for Emotion Prediction"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Emotions"),
    title="Speech Emotion Recognition - Inference Tool",
    description="Upload an audio file to test the CNN model. Displays the top 3 highest confidence emotions.",
    theme="default"
)

if __name__ == "__main__":
    # Launch server
    interface.launch(share=False)
