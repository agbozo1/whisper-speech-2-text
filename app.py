import streamlit as st
import torch
from torchaudio import transforms
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
#import os

# Streamlit app config
st.set_page_config(page_title="Romanian Speech Recognition", layout="centered")
st.title("🗣️ Romanian Speech-to-Text with Wav2Vec2")
st.write("Upload an audio file (wav, mp3, flac) and get the transcription in Romanian.")

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")

# Load Model and Processor
@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
    model = Wav2Vec2ForCTC.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()

# Resampler (48kHz -> 16kHz)
resampler = transforms.Resample(orig_freq=48_000, new_freq=16_000)

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

    # If not 48kHz, resample from actual sampling rate to 16kHz
    if sampling_rate != 48_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    
    # Resample the audio to 16kHz and convert to numpy
    speech = resampler(speech_array).squeeze().numpy()

    # Tokenize input
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    # Move to device
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Run inference
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    # Get prediction
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the prediction
    transcription = processor.batch_decode(predicted_ids)

    st.success("✅ Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:")
    st.markdown(f"> {transcription[0]}")

    # Clean up temp file
    #os.remove(tmp_file_path)
