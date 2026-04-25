import librosa
import joblib
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load models
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()
classifier = joblib.load("model/accent_classifier.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

def extract_features(file_path):
    """Extract Wav2Vec2 features from audio."""
    audio, _ = librosa.load(file_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def predict_accent(file_path):
    """Predict accent and confidence from audio file."""
    features = extract_features(file_path).reshape(1, -1)
    proba = classifier.predict_proba(features)[0]
    predicted_index = np.argmax(proba)
    predicted_accent = label_encoder.inverse_transform([predicted_index])[0]
    confidence = proba[predicted_index] * 100
    return predicted_accent, confidence
