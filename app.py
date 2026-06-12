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
st.write("Masukkan nilai indikator di bawah ini untuk melihat hasil estimasi model prediksi.")

# Fungsi untuk memuat model Scikit-Learn murni
@st.cache_resource
def load_model():
    # Menggunakan pure_model.pkl yang baru saja Anda buat
    with open("pure_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
    
    st.subheader("🔧 Parameter Indikator")
    
    # Membuat komponen input
    # Pastikan urutan variabel di sini SAMA PERSIS dengan urutan kolom saat Anda melatih model di Orange
    tahun = st.number_input("Tahun", min_value=2010, max_value=2040, value=2026, step=1)
    pdrb = st.number_input("PDRB (Produk Domestik Regional Bruto)", min_value=0.0, value=50000000.0, step=100000.0)
    inflasi = st.slider("Inflasi (%)", min_value=-5.0, max_value=25.0, value=3.5, step=0.1)
    jumlah_penerima = st.number_input("JUMLAH_PENERIMA (Penerima Subsidi)", min_value=0, value=2500, step=50)
    nilai_subsidi = st.number_input("NILAI_SUBSIDI (Total Anggaran)", min_value=0.0, value=150000000.0, step=500000.0)

    st.write("---")
    
    if st.button("🚀 Hitung Estimasi Tingkat Kemiskinan"):
        # Menyusun data input ke dalam format array 2D standar Scikit-Learn
        input_data = np.array([[tahun, pdrb, inflasi, jumlah_penerima, nilai_subsidi]])
        
        try:
            # Melakukan prediksi menggunakan model Scikit-Learn
            prediction = model.predict(input_data)
            
            # Menampilkan hasil output ke dashboard
            st.subheader("📈 Hasil Analisis Model")
            
            # Ekstraksi nilai
            predicted_value = prediction[0]
