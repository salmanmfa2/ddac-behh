import streamlit as st
import pickle
import numpy as np

# Pengaturan halaman utama
st.set_page_config(
    page_title="Dashboard Prediksi Kemiskinan",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Dashboard Prediksi Tingkat Kemiskinan")
st.write("Masukkan nilai indikator di bawah ini untuk melihat hasil estimasi.")

# Fungsi memuat model
@st.cache_resource
def load_model():
    with open("pure_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    # Load model ke memori
    model = load_model()
    
    st.subheader("🔧 Parameter Indikator")
    
    # Input dari user
    tahun = st.number_input("Tahun", min_value=2010, max_value=2040, value=2026, step=1)
    pdrb = st.number_input("PDRB", min_value=0.0, value=50000000.0, step=100000.0)
    inflasi = st.slider("Inflasi (%)", min_value=-5.0, max_value=25.0, value=3.5, step=0.1)
    jumlah_penerima = st.number_input("JUMLAH_PENERIMA", min_value=0, value=2500, step=50)
    nilai_subsidi = st.number_input("NILAI_SUBSIDI", min_value=0.0, value=150000000.0, step=500000.0)

    st.write("---")
    
    # Tombol prediksi
    if st.button("🚀 Hitung Estimasi Tingkat Kemiskinan"):
        # Susun data untuk model
        input_data = np.array([[tahun, pdrb, inflasi, jumlah_penerima, nilai_subsidi]])
        
        # Lakukan prediksi
        prediction = model.predict(input_data)
        
        # Tampilkan hasil
        st.subheader("📈 Hasil Analisis Model")
        st.metric(label="Prediksi Tingkat Kemiskinan", value=f"{prediction[0]:.2f} %")

except FileNotFoundError:
    st.error("File 'pure_model.pkl' tidak ditemukan. Pastikan file tersebut sudah ada di GitHub.")
except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")
