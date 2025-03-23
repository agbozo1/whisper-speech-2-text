import streamlit as st
import torch
import torchaudio
from torchaudio import transforms
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os

# Streamlit app config
st.set_page_config(page_title="Whisper Romanian Speech Recognition", layout="centered")
st.title("🗣️ Romanian Speech-to-Text with Whisper-small")
st.write("Upload an audio file (wav, mp3, flac) and get the transcription in Romanian.")

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")

# Load Model and Processor
@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()

# Upload Audio File
audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

if audio_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    # Display audio player
    st.audio(audio_file, format="audio/wav")

    # Load and preprocess the audio
    speech_array, sampling_rate = torchaudio.load(tmp_file_path)

    # Resample to 16kHz if necessary
    if sampling_rate != 16_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        resampled = resampler(speech_array)
    else:
        resampled = speech_array

    # Convert stereo to mono if necessary
    if resampled.shape[0] > 1:
        speech = resampled.mean(dim=0)
    else:
        speech = resampled.squeeze(0)

    # Processor requires float32
    input_features = processor(speech.numpy(), sampling_rate=16_000, return_tensors="pt").input_features

    # Move to device
    input_features = input_features.to(device)

    # Generate tokens
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="ro")

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    st.success("✅ Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:")
    st.markdown(f"> {transcription[0]}")

    # Optional: Clean up temp file
    os.remove(tmp_file_path)
