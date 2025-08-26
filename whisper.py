import streamlit as st
import torch
import torchaudio
from torchaudio import transforms
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import tempfile
import os
import numpy as np
import io
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment

st.set_page_config(page_title="Romanian Speech Recognition", layout="centered")
st.title("🗣️ Romanian Speech-to-Text Transcriber")

sound_source = st.radio(label="Select Audio Input Method:",
             options=['Upload', 'Record'])

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")


def load_audio(file_path, target_sr=16000):
    # Load with pydub (works for mp3, flac, wav)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)  # resample + mono
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    return samples, target_sr

@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()

audio_bytes = None  # Placeholder for audio data

if sound_source == "Upload":
    st.write("Upload an audio file (wav, mp3, flac):")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac"])
    if audio_file:
        audio_bytes = audio_file.read()

elif sound_source == "Record":
    st.write("Record your voice:")
    audio_bytes = audio_bytes = audio_recorder(
    text="🎙️ Record Your Voice",
    recording_color="#ff0000",  # red while recording
    neutral_color="#000000",    # black when idle
    icon_name="microphone"
)

# Process audio (if recorded or uploaded)
if audio_bytes:
    # Save audio bytes to temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        if sound_source == "Record":
            # Convert raw audio bytes to WAV if recorded
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            # Convert to 16-bit PCM format for saving
            audio_int16 = np.int16(audio_array * 32767)
            write_wav(tmp_file.name, 16000, audio_int16)
        else:
            # Directly save uploaded audio
            tmp_file.write(audio_bytes)

        tmp_file_path = tmp_file.name

    # Playback in Streamlit player
    st.audio(audio_bytes, format="audio/wav")

    # Load and preprocess audio
    #speech_array, sampling_rate = torchaudio.load(tmp_file_path)
    speech_array, sampling_rate = load_audio(tmp_file_path, target_sr=16000)

    # Resample to 16kHz if necessary
    if sampling_rate != 16_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        resampled = resampler(speech_array)
    else:
        resampled = speech_array

    # stereo to mono
    #if resampled.shape[0] > 1:
    #    speech = resampled.mean(dim=0)
    #else:
    #    speech = resampled.squeeze(0)
    # Handle mono / stereo consistently
    if resampled.dim() == 1:
        resampled = resampled.unsqueeze(0)

    speech = resampled.mean(dim=0).numpy()

    # Processor input
    input_features = processor(speech.numpy(), sampling_rate=16_000, return_tensors="pt").input_features
    input_features = input_features.to(device)

    # Whisper transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="ro")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    st.success("✅ Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:")
    st.markdown(f"> {transcription[0]}")

    # Clean up temp file
    os.remove(tmp_file_path)
