import streamlit as st
import pandas as pd
from datetime import datetime
from ttsModels import generate_speecht5_audio, generate_bark_audio
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Komparatif Model TTS", page_icon="🔬", layout="wide")

# Inisialisasi session state untuk menyimpan hasil terakhir
if 'latest_result' not in st.session_state:
    st.session_state.latest_result = None

# --- UI Sidebar ---
st.sidebar.title("🔬 Pengaturan Input")

# Fitur 2: Opsi Input Teks
st.sidebar.header("1. Teks Input")
predefined_texts = {
    "--- Ketik Manual ---": "",
    "Simple Sentence": "Hello, how are you doing today?",
    "Complex Sentence": "Despite the rain, I'm still planning to go for a walk, although I might take an umbrella.",
    "Question": "Do you think artificial intelligence will change the world?",
    "Exclamation": "Wow, this sounds incredibly realistic!",
    "Numbers & Acronyms": "The meeting is at 2:30 PM in room A-1, organized by NASA."
}
text_option = st.sidebar.selectbox("Pilih Teks Uji (opsional):", options=list(predefined_texts.keys()))

if text_option == "--- Ketik Manual ---":
    text_input_manual = st.sidebar.text_area("Atau ketik teks Anda di sini:", "The quick brown fox jumps over the lazy dog.", height=100)
    final_text_input = text_input_manual
else:
    final_text_input = predefined_texts[text_option]
    st.sidebar.text_area("Teks yang dipilih:", final_text_input, height=100, disabled=True)


# --- UI Halaman Utama ---
st.title("🔬 Analisis Komparatif Model Text-to-Speech")
st.markdown("Gunakan panel di sebelah kiri untuk mengatur input teks dan fitur model.")
st.info(f"**Teks yang akan diuji:** '{final_text_input}'")


col1, col2 = st.columns(2, gap="large")

# --- Kolom SpeechT5 ---
with col1:
    st.header("1. Microsoft SpeechT5")
    
    # Fitur 3: Pilihan Suara SpeechT5
    speaker_options = {
        "Wanita (Suara Jernih)": 7306,
        "Pria (Suara Berat)": 6799,
        "Wanita (Suara Ceria)": 236,
        "Pria (Suara Normal)": 5530
    }
    speaker_choice = st.selectbox("Pilih Jenis Suara:", options=list(speaker_options.keys()))
    speaker_id = speaker_options[speaker_choice]
    
    if st.button("Hasilkan Suara SpeechT5", use_container_width=True):
        if final_text_input:
            with st.spinner(f"Menghasilkan suara dengan '{speaker_choice}'..."):
                results = generate_speecht5_audio(final_text_input, speaker_id)
                st.session_state.latest_result = {"model": "SpeechT5", "text": final_text_input, "data": results}
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

# --- Kolom Bark ---
with col2:
    st.header("2. Suno's Bark")
    
    # Fitur 3: Fitur Unik Bark
    st.write("Tambahkan isyarat non-ucapan (opsional):")
    c1, c2, c3 = st.columns(3)
    laughter = c1.checkbox("[laughter]")
    sighs = c2.checkbox("[sighs]")
    clears_throat = c3.checkbox("[clears throat]")
    
    bark_text_input = final_text_input
    if laughter: bark_text_input += " [laughter]"
    if sighs: bark_text_input += " [sighs]"
    if clears_throat: bark_text_input += " [clears throat]"

    if st.button("Hasilkan Suara Bark", use_container_width=True):
        if bark_text_input:
            with st.spinner("Menghasilkan suara... (CPU, akan sangat lama)"):
                results = generate_bark_audio(bark_text_input)
                st.session_state.latest_result = {"model": "Bark", "text": bark_text_input, "data": results}
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

# --- Bagian Hasil dan Penyimpanan ---
st.markdown("---")
st.header("Hasil Generasi Terakhir")

if st.session_state.latest_result:
    res = st.session_state.latest_result
    st.subheader(f"Model: {res['model']}")
    
    # Fitur 1: Menampilkan Metrik
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Waktu Inferensi", f"{res['data']['inference_time']:.2f} s")
    m2.metric("Penggunaan RAM", f"{res['data']['ram_usage']:.2f} MB")
    m3.metric("Beban CPU", f"{res['data']['cpu_usage']:.2f} %")
    m4.metric("Ukuran Model", f"{res['data']['model_size_mb']:.2f} MB")
    m5.metric("Ukuran Audio", f"{res['data']['audio_size_kb']:.2f} KB")

    st.audio(res['data']['file_path'])

    # Fitur 4: Tombol Simpan Hasil
    if st.button("Simpan Hasil ke CSV"):
        # Siapkan data untuk disimpan
        log_data = {
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "model_name": [res['model']],
            "input_text": [res['text']],
            "inference_time": [res['data']['inference_time']],
            "cpu_usage": [res['data']['cpu_usage']],
            "ram_usage": [res['data']['ram_usage']],
        }
        df_new = pd.DataFrame(log_data)
        
        # Simpan ke CSV
        csv_file = "results.csv"
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
        else:
            df_new.to_csv(csv_file, index=False)
            
        st.success(f"Hasil berhasil disimpan ke dalam file `{csv_file}`!")
        # Tampilkan pratinjau data
        st.dataframe(pd.read_csv(csv_file).tail())

else:
    st.info("Tekan tombol 'Hasilkan Suara' pada salah satu model untuk melihat hasilnya di sini.")