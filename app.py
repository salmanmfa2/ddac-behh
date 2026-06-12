import streamlit as st
import pandas as pd
import pickle
import sklearn 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dashboard Prediksi Kemiskinan",
    page_icon="📊",
    layout="centered"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Loading the extracted lean scikit-learn model
    with open("model_kemiskinan.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- HEADER ---
st.title("📊 Prediksi Tingkat Kemiskinan")
st.markdown("""
Dashboard ini memprediksi **Tingkat Kemiskinan** berdasarkan indikator ekonomi makro dan penyaluran subsidi. 
Silakan masukkan parameter pada form di bawah ini.
""")
st.divider()

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("Parameter Input")
    
    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", min_value=2000, max_value=2050, value=2024, step=1)
        pdrb = st.number_input("PDRB (Rupiah)", min_value=0.0, value=1000000.0, step=10000.0)
        inflasi = st.number_input("Inflasi (%)", min_value=-10.0, max_value=100.0, value=3.5, step=0.1)
        
    with col2:
        jumlah_penerima = st.number_input("Jumlah Penerima", min_value=0, value=1000, step=10)
        nilai_subsidi = st.number_input("Nilai Subsidi", min_value=0.0, value=5000000.0, step=100000.0)

    submit_button = st.form_submit_button(label="Jalankan Prediksi", type="primary")

# --- PREDICTION LOGIC ---
if submit_button:
    # Constructing the DataFrame with exact feature names expected by the model
    input_data = pd.DataFrame({
        'Tahun': [tahun],
        'PDRB': [pdrb],
        'Inflasi': [inflasi],
        'JUMLAH_PENERIMA': [jumlah_penerima],
        'NILAI_SUBSIDI': [nilai_subsidi]
    })
    
    try:
        # Execute prediction
        prediction = model.predict(input_data)
        
        st.divider()
        st.success("✅ Prediksi Berhasil Dilakukan!")
        st.metric(label="Estimasi Tingkat Kemiskinan", value=str(prediction[0]))
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan pada saat prediksi: {e}")
        st.info("Catatan Analis: Pastikan tipe data input sejajar dengan format data training sebelumnya.")
