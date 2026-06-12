import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder

# Set page layout
st.set_page_config(page_title="Prediksi Tingkat Kemiskinan", layout="wide")

st.title("Prediksi Tingkat Kemiskinan dengan AdaBoost")
st.write("Aplikasi ini menggunakan model AdaBoost untuk memprediksi Tingkat Kemiskinan berdasarkan beberapa indikator.")

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

# Check if target and features exist in dataset
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {', '.join(missing_cols)}")
    st.stop()

# Preprocessing: Encode string categorical features to numeric
encoders = {}
X = df[features].copy()
y = df[target]

for col in features:
    le = LabelEncoder()
    # Convert to string to prevent errors with mixed data types
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train AdaBoost Model
model = AdaBoostRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# Sidebar for User Inputs
st.sidebar.header("Input Parameter")
user_input = {}

for col in features:
    # Get unique original string values for the selectbox
    unique_values = df[col].astype(str).unique().tolist()
    user_input[col] = st.sidebar.selectbox(f"Pilih {col}", unique_values)

# Make Prediction
st.header("Hasil Prediksi")

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Encode the user inputs using the fitted LabelEncoders
for col in features:
    input_df[col] = encoders[col].transform(input_df[col])

# Predict
prediction = model.predict(input_df)[0]
st.success(f"**Prediksi Tingkat Kemiskinan: {prediction:.2f}%**")

# Feature Importance
st.header("Feature Importance (Pentingnya Fitur)")
importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(features, importances, color='skyblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance dari Model AdaBoost')

# Invert Y axis to show the highest importance at the top
ax.invert_yaxis()  

# Display plot in Streamlit
st.pyplot(fig)

# Show raw data option
if st.checkbox("Tampilkan Data Asli"):
    st.write(df[[target] + features].head(10))
