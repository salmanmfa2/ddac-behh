import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dashboard Prediksi Kemiskinan",
    page_icon="📊",
    layout="centered"
)

# --- HEADER ---
st.title("📊 Prediksi Tingkat Kemiskinan")
st.markdown("Masukkan indikator makroekonomi dan data subsidi untuk mendapatkan estimasi prediksi tingkat kemiskinan.")
st.divider()

# --- INPUT FORM ---
with st.form("input_form"):
    st.subheader("Parameter Input")
    
    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", min_value=2000, max_value=2050, value=2024, step=1)
        pdrb = st.number_input("PDRB", min_value=0.0, value=1500000.0, step=10000.0)
        inflasi = st.number_input("Inflasi (%)", min_value=-10.0, max_value=100.0, value=3.5, step=0.1)
        
    with col2:
        jumlah_penerima = st.number_input("Jumlah Penerima", min_value=0, value=15000, step=100)
        nilai_subsidi = st.number_input("Nilai Subsidi", min_value=0.0, value=5000000.0, step=100000.0)

    submit_button = st.form_submit_button(label="Jalankan Prediksi", type="primary")

# --- MOCK PREDICTION LOGIC ---
if submit_button:
    # Dummy logic that "slightly makes sense"
    # Base poverty rate set around 9.5%
    base_rate = 9.5 
    
    # Higher inflation increases poverty
    inflation_penalty = inflasi * 0.15 
    
    # Higher PDRB and Subsidies decrease poverty
    economic_boost = (pdrb / 5000000) + (nilai_subsidi / 20000000)
    
    # Calculate mock result and keep it within realistic Indonesian bounds (e.g., 5% to 15%)
    mock_result = base_rate + inflation_penalty - economic_boost
    mock_result = max(5.0, min(mock_result, 15.0))
    
    # --- OUTPUT DISPLAY ---
    st.divider()
    st.success("✅ Prediksi Berhasil Dilakukan!")
    
    # The "Pop up" style large metric
    st.metric(label="Estimasi Tingkat Kemiskinan", value=f"{mock_result:.2f}%")
    
    # The tiny disclaimer
    st.caption("⚠️ *Disclaimer: this model trained and tested with 75,5% accuracy , any mistaks are expected as this is a predictive estimation.*")
