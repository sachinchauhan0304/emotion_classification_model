# Speech Emotion Recognition Web App using GRU + Attention

## 🔍 Project Overview

This project involves the development and deployment of a Speech Emotion Recognition (SER) model that identifies human emotions from audio signals using MFCC features. The backend leverages a GRU-based neural network enhanced with an attention mechanism. The app accepts .wav audio inputs and classifies them into one of eight predefined emotional categories.

## 🎯 Objectives

* Analyze and detect emotions in speech through deep learning.
* Build an easy-to-use web interface using Streamlit.
* Deploy the app seamlessly on Streamlit Cloud.

## 🧠 Dataset: RAVDESS

* Source: [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
* Subset: Only **speech** files (Modality code = `01`) used.
* Classes: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

## 🧹 Data Preprocessing

1. **File Parsing**: Traverse RAVDESS folders to extract file paths and emotion labels.
2. **Audio Features**: Extract 40-dimensional MFCCs per frame with `librosa`.
3. **Padding**: All MFCCs padded/truncated to a fixed length of 174 frames for consistency.
4. **Label Encoding**: Emotions encoded using `LabelEncoder` from `sklearn`.

## 🧱 Model Architecture

* **Input**: `(174, 40)` MFCC features
* **Layer 1**: Bidirectional GRU (128 units)
* **Layer 2**: Attention Layer (custom)
* **Dense**: Fully connected layer for classification
* **Activation**: `softmax`

> ✅ The model is trained using categorical cross-entropy and Adam optimizer.

## 📊 Performance Metrics

After tuning and training:

* ✅ **Macro F1 Score**: **84.32%**
* ✅ **Overall Accuracy**: **85%**
* Per-class metrics:

  * `angry`: F1 = 0.92
  * `calm`: F1 = 0.88
  * `happy`: F1 = 0.84
  * `surprised`: F1 = 0.91
  * `neutral`: F1 = 0.74

## 🌐 Web Application (Streamlit)

The app allows users to:

* Upload `.wav` files
* Listen to audio playback
* View predicted emotion

### Key Features

* Clean UI with `st.audio()` and emotion output
* Loads pre-trained GRU-Attention model and label encoder (`.h5` and `.pkl`)
* Uses `librosa` for MFCC extraction

## 🛠️ Setup & Deployment

### Local

```bash
# Clone repo
https://github.com/sachinchauhan0304/emotion_classification.git

# Create virtual env
python -m venv mars
source mars/bin/activate  # or .\mars\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

### Streamlit Cloud

* ✅ Python 3.9 runtime (set automatically by Streamlit)
* ✅ Ensure `emotion_gru_attention_model.h5` and `label_encoder.pkl` are in repo
* ✅ Correct `requirements.txt` (compatible with both `tensorflow==2.11.0` and `streamlit==1.25.0`)

## 📁 Project Structure

```
emotion-classification-app/
├── app.py
├── emotion_gru_attention_model.h5
├── label_encoder.pkl
├── requirements.txt
├── test.py
└── README.md
```

## 💡 Future Improvements

* Use larger and more diverse datasets
* Add multilingual support
* Include live microphone input
* Try transformer-based architectures (e.g., Wav2Vec, HuBERT)

---

**Author**: [Sachin Chauhan](https://github.com/sachinchauhan0304)

**License**: MIT
