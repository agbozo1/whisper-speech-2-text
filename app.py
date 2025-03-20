import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os
import soundfile as sf

# Set Streamlit page configuration
st.set_page_config(page_title="Whisper Speech Recognition", layout="centered")

st.title("🎙️ Whisper Speech-to-Text Transcription")
st.write("Upload an audio file and get a transcription using Whisper Large V3 Turbo.")

# Set device and torch dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor once
@st.cache_resource(show_spinner=True)
def load_model():
    model_id = "openai/whisper-large-v3-turbo"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe

pipe = load_model()

# File uploader
audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

if audio_file is not None:
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        temp_file_path = tmp_file.name
    
    st.audio(audio_file, format="audio/wav")

    st.info("Transcribing the audio... Please wait.")

    # Load and process audio using soundfile
    audio_input, sample_rate = sf.read(temp_file_path)

    # Transcribe the audio
    result = pipe({"array": audio_input, "sampling_rate": sample_rate})

    # Show result
    st.success("Transcription Completed!")
    st.markdown(f"### 📝 Transcribed Text:\n\n{result['text']}")

    # Clean up temp file
    os.remove(temp_file_path)
