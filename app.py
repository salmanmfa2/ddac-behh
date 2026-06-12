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
        # 1. Buat array kosong dengan 33 kolom (sesuai kebutuhan model)
        # Kita isi dengan 0 sebagai nilai default untuk 28 variabel dummy/fitur tambahan
        full_input = np.zeros((1, 33)) 
        
        # 2. Masukkan 5 data input kita ke kolom yang tepat
        # Catatan: Jika urutannya salah, hasil prediksi akan meleset. 
        # Jika hasil prediksi aneh, kita mungkin perlu menyesuaikan indeksnya.
        full_input[0, 0] = tahun
        full_input[0, 1] = pdrb
        full_input[0, 2] = inflasi
        full_input[0, 3] = jumlah_penerima
        full_input[0, 4] = nilai_subsidi
        
        # 3. Lakukan prediksi
        prediction = model.predict(full_input)
        
        # 4. Tampilkan hasil
        st.subheader("📈 Hasil Analisis Model")
        st.metric(label="Prediksi Tingkat Kemiskinan", value=f"{prediction[0]:.2f} %")

except FileNotFoundError:
    st.error("File 'pure_model.pkl' tidak ditemukan. Pastikan file tersebut sudah ada di GitHub.")
except Exception as e:
    st.error(f"Terjadi kesalahan sistem: {e}")
