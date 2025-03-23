import streamlit as st
import torch
import torchaudio
from torchaudio import transforms
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os

st.set_page_config(page_title="Romanian Speech Recognition", layout="centered")
st.title("🗣️ Romanian Speech-to-Text Transcriber")

sound_source = st.radio(label="Upload Audio / Record Audio",
             options=['Upload','Record'])

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")

@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()

if sound_source == "Upload":
    st.write("Upload an audio file (wav, mp3, flac) and get the transcription in Romanian.")
    audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac"])
elif sound_source == "Record":
    st.write("Record Now!")
    audio_file = st.audio_input("Record a voice message")


if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    st.audio(audio_file, format="audio/wav")

    speech_array, sampling_rate = torchaudio.load(tmp_file_path)

    # Resample to 16kHz (if necessary)
    if sampling_rate != 16_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        resampled = resampler(speech_array)
    else:
        resampled = speech_array

    # stereo to mono (if necessary)
    if resampled.shape[0] > 1:
        speech = resampled.mean(dim=0)
    else:
        speech = resampled.squeeze(0)

    input_features = processor(speech.numpy(), sampling_rate=16_000, return_tensors="pt").input_features

    input_features = input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="ro")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    st.success("✅ Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:")
    st.markdown(f"> {transcription[0]}")

    os.remove(tmp_file_path)
