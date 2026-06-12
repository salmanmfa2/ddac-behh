import streamlit as st
import pickle
import Orange
import numpy as np

# Pengaturan halaman utama
st.set_page_config(
    page_title="Dashboard Prediksi Kemiskinan",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Dashboard Prediksi Tingkat Kemiskinan")
st.write("Masukkan nilai indikator di bawah ini untuk melihat hasil estimasi model prediksi.")

# Fungsi untuk memuat model Orange secara aman dengan caching
@st.cache_resource
def load_predictive_model():
    # File .pkcls dibaca langsung menggunakan modul pickle bawaan Python
    with open("model.pkcls", "rb") as file:
        return pickle.load(file)

try:
    model = load_predictive_model()
    
    st.subheader("🔧 Parameter Indikator")
    
    # Membuat komponen input untuk setiap fitur sesuai tipe datanya
    tahun = st.number_input("Tahun", min_value=2010, max_value=2040, value=2026, step=1)
    pdrb = st.number_input("PDRB (Produk Domestik Regional Bruto)", min_value=0.0, value=50000000.0, step=100000.0)
    inflasi = st.slider("Inflasi (%)", min_value=-5.0, max_value=25.0, value=3.5, step=0.1)
    jumlah_penerima = st.number_input("JUMLAH_PENERIMA (Penerima Subsidi)", min_value=0, value=2500, step=50)
    nilai_subsidi = st.number_input("NILAI_SUBSIDI (Total Anggaran)", min_value=0.0, value=150000000.0, step=500000.0)

    st.write("---")
    
    if st.button("🚀 Hitung Estimasi Tingkat Kemiskinan"):
        # Menyusun data input ke dalam format matriks/list 2D
        # Catatan: Kolom 'TAHUN' hasil merge otomatis diabaikan sesuai instruksi Anda
        raw_values = [[tahun, pdrb, inflasi, jumlah_penerima, nilai_subsidi]]
        
        try:
            # 'Magic' yang sesungguhnya: Membuat objek Orange Table menggunakan Domain asli dari model
            # Ini memastikan data dipetakan ke kolom yang benar secara internal
            input_table = Orange.data.Table(model.domain, raw_values)
            
            # Melakukan prediksi
            prediction = model(input_table)
            
            # Menampilkan hasil output ke dashboard
            st.subheader("📈 Hasil Analisis Model")
            
            # Ekstraksi nilai numerik dari output prediksi Orange
            if hasattr(prediction[0], '__iter__'):
                predicted_value = prediction[0][0]
            else:
                predicted_value = prediction[0]
                
            st.metric(
                label="Prediksi Tingkat Kemiskinan", 
                value=f"{predicted_value:.2f} %"
            )
            
        except Exception as error:
            st.error(f"Gagal memproses input ke dalam domain model. Pastikan struktur data sesuai. Detail: {error}")

except FileNotFoundError:
    st.error("File 'model.pkcls' tidak ditemukan di direktori aktif. Pastikan file model berada dalam satu folder dengan script 'app.py' ini.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat dashboard: {e}")
