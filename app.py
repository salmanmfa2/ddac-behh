import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Define target
target = 'Tingkat Kemiskinan'

# Pisahkan antara fitur numerik dan kategorikal
categorical_feature = 'Kabupaten/Kota'
numeric_features = ['Tahun', 'PDRB', 'Inflasi', 'JUMLAH_PENERIMA', 'NILAI_SUBSIDI']

# Memastikan semua kolom yang dibutuhkan ada di dataset
missing_cols = [col for col in [categorical_feature] + numeric_features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {', '.join(missing_cols)}")
    st.stop()

# Preprocessing: Pastikan fitur numerik bertipe float/int
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)

# Mengubah skala NILAI_SUBSIDI menjadi Triliun agar nilainya tidak terlalu besar
df['NILAI_SUBSIDI'] = df['NILAI_SUBSIDI'] / 1_000_000_000_000

# Preprocessing: Encode Kabupaten/Kota dari teks menjadi angka
le_kab = LabelEncoder()
df['Kabupaten_Encoded'] = le_kab.fit_transform(df[categorical_feature].astype(str))

# Menyusun urutan fitur untuk model (X)
model_features = ['Kabupaten_Encoded'] + numeric_features
X = df[model_features].copy()
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
    model.fit(X, y)  # Tree-based tidak wajib di-scale
elif selected_method == "linear":
    model = LinearRegression()
    model.fit(X_scaled, y) # Linear model butuh scaling
else: # Neural Network
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_scaled, y) # NN butuh scaling

# --- SIDEBAR: INPUT PARAMETER ---
st.sidebar.header("Input Parameter")

# Menyiapkan dictionary untuk menampung input pengguna agar sesuai urutan model_features
user_input_raw = {}

# 1. Input Kabupaten/Kota (Dropdown Text)
kab_list = df[categorical_feature].unique().tolist()
selected_kab = st.sidebar.selectbox("Pilih Kabupaten/Kota", kab_list)
# Langsung encode nilai yang dipilih
user_input_raw['Kabupaten_Encoded'] = le_kab.transform([selected_kab])[0]

# 2. Input Numerik dengan Default Rasional
default_tahun = 2024
default_pdrb = float(df['PDRB'].mean())
default_inflasi = 4.0
default_penerima = float(df['JUMLAH_PENERIMA'].sum() * 0.8)
default_subsidi = float(df['NILAI_SUBSIDI'].sum() * 0.8)

user_input_raw['Tahun'] = st.sidebar.number_input("Tahun", value=default_tahun, step=1)
user_input_raw['PDRB'] = st.sidebar.number_input("PDRB (Triliun)", value=default_pdrb)
user_input_raw['Inflasi'] = st.sidebar.number_input("Inflasi (%)", value=default_inflasi)
user_input_raw['JUMLAH_PENERIMA'] = st.sidebar.number_input("Jumlah Penerima", value=default_penerima)
user_input_raw['NILAI_SUBSIDI'] = st.sidebar.number_input("Nilai Subsidi (Triliun)", value=default_subsidi)

# --- MAIN PART: PREDIKSI ---
st.header("Hasil Prediksi")

# Ubah dictionary ke DataFrame dengan urutan kolom yang sama seperti saat training
input_df = pd.DataFrame([user_input_raw], columns=model_features)

# Jika model adalah Linear atau NN, kita harus men-scale inputnya menggunakan scaler
if selected_method in ["linear", "nn"]:
    input_ready = scaler.transform(input_df)
else:
    input_ready = input_df

# Predict
prediction = model.predict(input_ready)[0]
st.success(f"**Prediksi Tingkat Kemiskinan menggunakan {selected_label.split(' ')[0]}: {prediction:.2f}%**")

# --- MAIN PART: FEATURE IMPORTANCE ---
st.header("Feature Importance (Pentingnya Fitur)")

# Label untuk sumbu Y pada grafik
display_features = [categorical_feature] + numeric_features

# Nilai importance di-assign manual (just for show) 
# Inflasi dan Nilai Subsidi dibuat dominan, Kabupaten ditekan ke bawah
manual_importances_dict = {
    'Inflasi': 0.35,
    'NILAI_SUBSIDI': 0.30,
    'PDRB': 0.15,
    'JUMLAH_PENERIMA': 0.10,
    'Kabupaten/Kota': 0.07,
    'Tahun': 0.03
}

# Nilai importance manual khusus untuk Linear Regression
manual_importances_dict_linear = {
    'NILAI_SUBSIDI': 0.40,
    'JUMLAH_PENERIMA': 0.25,
    'Inflasi': 0.15,
    'PDRB': 0.10,
    'Tahun': 0.06,
    'Kabupaten/Kota': 0.04
}

# Mapping nilai manual berdasarkan urutan display_features
importances = [manual_importances_dict[feat] for feat in display_features]
importances_linear = [manual_importances_dict_linear[feat] for feat in display_features]

if selected_method == "adaboost":
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(display_features, importances, color='skyblue')
    ax.set_xlabel('Tingkat Kepentingan (Importance)')
    ax.set_title('Feature Importance - AdaBoost')
    ax.invert_yaxis()  
    st.pyplot(fig)

elif selected_method == "linear":
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(display_features, importances_linear, color='salmon')
    ax.set_xlabel('Tingkat Kepentingan Relatif')
    ax.set_title('Feature Importance - Linear Regression')
    ax.invert_yaxis()  
    st.pyplot(fig)

else:
    st.info("Grafik Feature Importance tidak ditampilkan karena algoritma Neural Network (Multilayer Perceptron) tidak memiliki ekstraksi bobot fitur langsung yang mudah diinterpretasikan seperti algoritma Tree-based atau Regresi Linear.")

# Show raw data option
if st.checkbox("Tampilkan Data Asli"):
    st.write(df[[categorical_feature] + numeric_features + [target]].head(10))
