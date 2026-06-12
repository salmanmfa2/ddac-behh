import streamlit as st
import pandas as pd
import pickle

# Load model
with open("pure_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar navigation
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Go to:", ["Prediction", "Batch Upload", "About"])

# --- Prediction Page ---
if page == "Prediction":
    st.title("📈 Tingkat Kemiskinan Predictor")

    st.markdown("Enter the economic indicators below to predict **Tingkat Kemiskinan**.")

    col1, col2 = st.columns(2)

    with col1:
        tahun = st.number_input("Tahun", min_value=2000, max_value=2100, step=1)
        pdrb = st.number_input("PDRB", min_value=0.0, format="%.2f")

    with col2:
        inflasi = st.number_input("Inflasi (%)", min_value=0.0, format="%.2f")
        jumlah_penerima = st.number_input("Jumlah Penerima", min_value=0)
        nilai_subsidi = st.number_input("Nilai Subsidi", min_value=0.0, format="%.2f")

    input_data = pd.DataFrame({
        "Tahun": [tahun],
        "PDRB": [pdrb],
        "Inflasi": [inflasi],
        "JUMLAH_PENERIMA": [jumlah_penerima],
        "NILAI_SUBSIDI": [nilai_subsidi]
    })

    if st.button("🔮 Predict"):
        prediction = model.predict(input_data)[0]
        st.metric(label="Predicted Tingkat Kemiskinan", value=f"{prediction:.2f}")

# --- Batch Upload Page ---
elif page == "Batch Upload":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV file with the required columns to predict multiple rows at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        predictions = model.predict(df)
        df["Predicted Tingkat Kemiskinan"] = predictions
        st.success("✅ Predictions completed!")
        st.write(df)

        st.download_button(
            label="📥 Download Results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

# --- About Page ---
else:
    st.title("ℹ️ About This App")
    st.markdown("""
    This dashboard predicts **Tingkat Kemiskinan** using a trained machine learning model.
    
    **Features used:**
    - Tahun  
    - PDRB  
    - Inflasi  
    - Jumlah Penerima  
    - Nilai Subsidi  

    Built with **Streamlit** and deployed via **Streamlit Cloud**.
    """)
