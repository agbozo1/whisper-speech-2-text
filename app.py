import streamlit as st
import torch
from torchaudio import transforms
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
from pydub import AudioSegment
import numpy as np 

st.set_page_config(page_title="Romanian Speech Recognition", layout="centered")
st.title("🗣️ Romanian Speech-to-Text with Wav2Vec2")
st.write("Upload an audio file (wav, mp3, flac) and get the transcription in Romanian.")


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
    processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
    model = Wav2Vec2ForCTC.from_pretrained("anton-l/wav2vec2-large-xlsr-53-romanian")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()


resampler = transforms.Resample(orig_freq=48_000, new_freq=16_000)


audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    st.audio(audio_file, format="audio/wav")

    speech_array, sampling_rate = torchaudio.load(tmp_file_path)


    if sampling_rate != 48_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    
    #speech_array, sampling_rate = torchaudio.load(tmp_file_path)
    speech_array, sampling_rate = load_audio(tmp_file_path, target_sr=16000)

    if sampling_rate != 48_000:
        resampler = transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)


    resampled = resampler(speech_array)

    if resampled.shape[0] > 1:
        speech = resampled.mean(dim=0).numpy()
    else:
        speech = resampled.squeeze().numpy()

    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)

    st.success("✅ Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:")
    st.markdown(f"> {transcription[0]}")

    #os.remove(tmp_file_path)
