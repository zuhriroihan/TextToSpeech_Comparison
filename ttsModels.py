import os
import torch
import numpy as np
import glob
import time
from zipfile import ZipFile
import pyarrow.feather as feather
import soundfile as sf
import streamlit as st
import psutil
import requests
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    AutoProcessor,
    BarkModel,
)

# Proses yang berjalan saat ini untuk memonitor resource
p = psutil.Process(os.getpid())

@st.cache_resource
def load_speecht5_components():
    """
    Memuat semua komponen SpeechT5, termasuk mengunduh dan memproses
    speaker embedding secara manual.
    """
    print("Memuat komponen utama SpeechT5...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    print("Komponen utama SpeechT5 berhasil dimuat.")

    # --- Logika Pemuatan Speaker Embedding (Metode ZIP yang Benar) ---
    zip_filename = "spkrec-xvect.zip"
    zip_url = "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/main/spkrec-xvect.zip"
    extract_dir = "spkrec-xvect"
    
    if not os.path.exists(zip_filename):
        print(f"File '{zip_filename}' tidak ditemukan. Mengunduh...")
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            with open(zip_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Unduhan ZIP selesai.")
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal mengunduh file ZIP: {e}")
            return None, None, None, None, 0
    
    if not os.path.isdir(extract_dir):
        print(f"Mengekstrak '{zip_filename}'...")
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Ekstraksi selesai.")

    try:
        npy_files = sorted(glob.glob(os.path.join(extract_dir, '*.npy')))
        if not npy_files:
            st.error("Tidak ada file .npy yang ditemukan setelah ekstraksi.")
            return None, None, None, None, 0
        
        embeddings_list = [np.load(f) for f in npy_files]
        print(f"Berhasil memuat {len(embeddings_list)} data speaker dari file .npy.")
        # Untuk ukuran model, kita akan gunakan nilai statis karena menghitungnya dari cache bisa rumit
        model_size_mb = 485 
        return processor, model, vocoder, embeddings_list, model_size_mb
        
    except Exception as e:
        st.error(f"Gagal memuat file .npy: {e}")
        return None, None, None, None, 0

@st.cache_resource
def load_bark_model():
    """Memuat model dan processor Bark."""
    print("Memuat komponen Bark...")
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small").to("cpu")
    print("Komponen Bark berhasil dimuat.")
    # Ukuran statis untuk model Bark small
    model_size_mb = 1430 
    return processor, model, model_size_mb

def generate_speecht5_audio(text_input, speaker_id):
    """Menghasilkan audio menggunakan SpeechT5 dan mengukur metrik."""
    processor, model, vocoder, embeddings_list, model_size_mb = load_speecht5_components()
    
    if processor is None: return None # Berhenti jika loading gagal

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    vocoder.to(device)

    mem_before = p.memory_info().rss / (1024 * 1024)
    cpu_before = p.cpu_percent(interval=None)
    start_time = time.time()

    inputs = processor(text=text_input, return_tensors="pt").to(device)
    speaker_embeddings = torch.tensor(embeddings_list[speaker_id]).unsqueeze(0).to(device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
    
    end_time = time.time()
    mem_after = p.memory_info().rss / (1024 * 1024)
    cpu_after = p.cpu_percent(interval=None)

    output_filename = "output_speecht5.wav"
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    
    results = {
        "file_path": output_filename,
        "inference_time": end_time - start_time,
        "cpu_usage": cpu_after - cpu_before,
        "ram_usage": mem_after - mem_before,
        "model_size_mb": model_size_mb,
        "audio_size_kb": os.path.getsize(output_filename) / 1024
    }
    return results

def generate_bark_audio(text_input, voice_preset="v2/en_speaker_6"):
    """Menghasilkan audio menggunakan Bark dan mengukur metrik."""
    processor, model, model_size_mb = load_bark_model()

    if processor is None: return None

    mem_before = p.memory_info().rss / (1024 * 1024)
    cpu_before = p.cpu_percent(interval=None)
    start_time = time.time()

    inputs = processor(text=text_input, voice_preset=voice_preset, return_tensors="pt")
    speech = model.generate(**inputs, do_sample=True, fine_temperature=0.4, coarse_temperature=0.8)

    end_time = time.time()
    mem_after = p.memory_info().rss / (1024 * 1024)
    cpu_after = p.cpu_percent(interval=None)

    output_filename = "output_bark.wav"
    sampling_rate = model.generation_config.sample_rate
    sf.write(output_filename, speech.cpu().numpy().squeeze(), samplerate=sampling_rate)
    
    results = {
        "file_path": output_filename,
        "inference_time": end_time - start_time,
        "cpu_usage": cpu_after - cpu_before,
        "ram_usage": mem_after - mem_before,
        "model_size_mb": model_size_mb,
        "audio_size_kb": os.path.getsize(output_filename) / 1024
    }
    return results