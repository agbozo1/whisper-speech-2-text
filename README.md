# 🗣️ Romanian Speech-to-Text

A simple Streamlit app for converting Romanian speech to text using **Wav2Vec2**.

## 🚀 Features
- Upload `.wav`, `.mp3`, or `.flac` audio files  
- Automatic transcription into Romanian  
- Runs on CPU or GPU  

## 🔧 Tech Stack
- [Streamlit](https://streamlit.io/) – UI  
- [Hugging Face Transformers](https://huggingface.co/) – Speech model  
- [PyTorch](https://pytorch.org/)  
- [pydub](https://github.com/jiaaro/pydub) – audio loading  

## ▶️ Run Locally
```bash
git clone https://github.com/your-username/whisper-speech-2-text.git
cd whisper-speech-2-text
pip install -r requirements.txt
streamlit run app.py
