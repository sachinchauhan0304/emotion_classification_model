import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys

# Load model & label encoder
model = tf.keras.models.load_model("emotion_gru_attention_model.h5", compile=False)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def preprocess_audio(file_path, n_mfcc=40, max_pad_len=174):
    y, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = np.transpose(mfcc[np.newaxis, ...], (0, 2, 1))
    return mfcc

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    features = preprocess_audio(audio_path)
    prediction = model.predict(features)
    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"🎤 Predicted Emotion: {pred_label[0]}")
