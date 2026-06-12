import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Set page layout
st.set_page_config(page_title="Prediksi Tingkat Kemiskinan", layout="wide")

st.title("Prediksi Tingkat Kemiskinan")
st.write("Aplikasi ini memprediksi Tingkat Kemiskinan berdasarkan beberapa indikator menggunakan metode Machine Learning pilihan Anda.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("final_Data.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'final_Data.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.")
    st.stop()

# Define features and target
target = 'Tingkat Kemiskinan'
features = ['Tahun', 'PDRB', 'Inflasi', 'JUMLAH_PENERIMA', 'NILAI_SUBSIDI']

# Memastikan semua kolom fitur bertipe numerik. 
# Jika ada sisa format teks/kategori dari dataset sebelumnya, akan dikonversi menjadi NaN lalu diisi 0 agar aplikasi tidak crash.
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)

X = df[features].copy()
y = df[target]

# Scaling data untuk Linear Regression dan Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SIDEBAR: PEMILIHAN MODEL ---
st.sidebar.header("Pengaturan Model")

model_options = {
    "AdaBoost 75,5%": "adaboost",
    "Linear Regression 94,54% (Caution Overfitted Model)": "linear",
    "Neural Network 66,43%": "nn"
}

selected_label = st.sidebar.radio(
    "Pilih Metode Machine Learning:", 
    list(model_options.keys())
)
selected_method = model_options[selected_label]

# Training Model Berdasarkan Pilihan
if selected_method == "adaboost":
    model = AdaBoostRegressor(n_estimators=50, random_state=42)
    model.fit(X, y) # Tree-based menggunakan data asli
elif selected_method == "linear":
    model = LinearRegression()
    model.fit(X_scaled, y) # Linear model butuh scaling
else: # Neural Network
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_scaled, y) # NN butuh scaling

# --- SIDEBAR: INPUT PARAMETER DENGAN PRE-INPUT RASIONAL ---
st.sidebar.header("Input Parameter")

# Menghitung default values sesuai permintaan
default_tahun = 2024
default_pdrb = float(df['PDRB'].mean())
default_inflasi = 4.0
default_penerima = float(df['JUMLAH_PENERIMA'].mean() * 0.8)
default_subsidi = float(df['NILAI_SUBSIDI'].mean() * 0.8)

user_input = {}

user_input['Tahun'] = st.sidebar.number_input("Tahun", value=default_tahun, step=1)
user_input['PDRB'] = st.sidebar.number_input("PDRB", value=default_pdrb)
user_input['Inflasi'] = st.sidebar.number_input("Inflasi (%)", value=default_inflasi)
user_input['JUMLAH_PENERIMA'] = st.sidebar.number_input("Jumlah Penerima", value=default_penerima)
user_input['NILAI_SUBSIDI'] = st.sidebar.number_input("Nilai Subsidi", value=default_subsidi)

# --- MAIN PART: PREDIKSI ---
st.header("Hasil Prediksi")

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Jika model adalah Linear atau NN, kita men-scale inputnya menggunakan scaler yang sudah di-fit dengan data latih
if selected_method in ["linear", "nn"]:
    input_ready = scaler.transform(input_df)
else:
    input_ready = input_df

# Predict
prediction = model.predict(input_ready)[0]
st.success(f"**Prediksi Tingkat Kemiskinan menggunakan {selected_label.split(' ')[0]}: {prediction:.2f}%**")

# --- MAIN PART: FEATURE IMPORTANCE ---
st.header("Feature Importance")

if selected_method == "adaboost":
    importances = model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(features, importances, color='skyblue')
    ax.set_xlabel('Tingkat Kepentingan (Importance)')
    ax.set_title('Feature Importance - AdaBoost')
    ax.invert_yaxis()  
    st.pyplot(fig)

elif selected_method == "linear":
    importances = np.abs(model.coef_)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(features, importances, color='salmon')
    ax.set_xlabel('Nilai Absolut Koefisien')
    ax.set_title('Feature Importance (Koefisien) - Linear Regression')
    ax.invert_yaxis()  
    st.pyplot(fig)

else:
    st.info("Grafik Feature Importance tidak ditampilkan karena algoritma Neural Network tidak menghasilkan bobot fitur yang dapat divisualisasikan secara langsung seperti algoritma Tree-based atau Regresi Linear.")

# Show raw data option
if st.checkbox("Tampilkan Data Asli"):
    st.write(df[[target] + features].head(10))
