import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL (safe loading)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("pure_model.pkl")

model = load_model()

# =========================
# UI
# =========================
st.title("📊 Prediksi Tingkat Kemiskinan")

st.markdown("Masukkan parameter:")

col1, col2 = st.columns(2)

with col1:
    tahun = st.number_input("Tahun", 2000, 2100, 2023)
    pdrb = st.number_input("PDRB", min_value=0.0)

with col2:
    inflasi = st.number_input("Inflasi (%)", min_value=0.0)
    jumlah_penerima = st.number_input("Jumlah Penerima", min_value=0)
    nilai_subsidi = st.number_input("Nilai Subsidi", min_value=0.0)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Prediksi"):
    input_df = pd.DataFrame({
        'Tahun': [tahun],
        'PDRB': [pdrb],
        'Inflasi': [inflasi],
        'JUMLAH_PENERIMA': [jumlah_penerima],
        'NILAI_SUBSIDI': [nilai_subsidi]
    })

    try:
        pred = model.predict(input_df)
        st.success(f"Hasil Prediksi: {pred[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
