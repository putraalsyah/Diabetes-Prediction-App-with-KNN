"""
===============================================================================
APLIKASI WEB PREDIKSI DIABETES MENGGUNAKAN K-NEAREST NEIGHBORS (KNN)
===============================================================================

Nama    : Putra Aliansyah
NIM     : 301230041
Kelas   : IF 5A
Dosen   : Mohammad Bayu Anggara, S.Kom., M.Kom.

File    : app.py (Flask Backend)
Deskripsi: Web application untuk prediksi risiko diabetes menggunakan model KNN
===============================================================================
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

model = None
scaler = None
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# ============================================================================
# FUNGSI UNTUK TRAINING DAN LOAD MODEL
# ============================================================================

def train_and_save_model():
    """
    Fungsi untuk training model KNN dan menyimpannya sebagai pickle file.
    Fungsi ini akan otomatis dipanggil jika model belum ada.
    """
    print("Training model KNN...")
    
    # Load dataset (pastikan file diabetes.csv ada di folder yang sama)
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("ERROR: File diabetes.csv tidak ditemukan!")
        print("Silakan download dataset dari: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        return False
    
    # Handling zero values
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_with_zero:
        df[col] = df[col].replace(0, np.nan)
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
    
    # Pemisahan fitur dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training model KNN dengan K=8 (optimal dari penelitian)
    knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean', weights='uniform')
    knn.fit(X_train_scaled, y_train)
    
    # Evaluasi
    accuracy = knn.score(X_test_scaled, y_test)
    print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
    
    # Save model dan scaler
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model dan scaler berhasil disimpan!")
    return True

def load_model():
    """
    Fungsi untuk load model dan scaler yang sudah di-training.
    Jika model belum ada, akan melakukan training terlebih dahulu.
    """
    global model, scaler
    
    # Cek apakah model dan scaler sudah ada
    if os.path.exists('knn_model.pkl') and os.path.exists('scaler.pkl'):
        try:
            with open('knn_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("Model dan scaler berhasil dimuat!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Melakukan training ulang...")
            return train_and_save_model()
    else:
        print("Model belum ada. Melakukan training...")
        return train_and_save_model()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """
    Route untuk halaman utama
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route untuk melakukan prediksi diabetes
    Input: JSON dengan 8 fitur
    Output: JSON dengan hasil prediksi dan probabilitas
    """
    try:
        # Ambil data dari request
        data = request.get_json()
        
        # Ekstrak fitur
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodPressure', 0)),
            float(data.get('skinThickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('diabetesPedigreeFunction', 0)),
            float(data.get('age', 0))
        ]
        
        # Validasi input
        if features[1] <= 0 or features[5] <= 0 or features[7] <= 0:
            return jsonify({
                'error': 'Glucose, BMI, dan Age harus lebih dari 0!'
            }), 400
        
        # Convert ke numpy array dan reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scaling
        features_scaled = scaler.transform(features_array)
        
        # Prediksi
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Format hasil
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Diabetes Positif' if prediction == 1 else 'Tidak Diabetes',
            'probability_no_diabetes': float(probabilities[0] * 100),
            'probability_diabetes': float(probabilities[1] * 100),
            'risk_level': get_risk_level(probabilities[1]),
            'recommendation': get_recommendation(prediction, probabilities[1], features)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Terjadi kesalahan: {str(e)}'
        }), 500

def get_risk_level(prob_diabetes):
    """
    Menentukan level risiko berdasarkan probabilitas
    """
    if prob_diabetes < 0.3:
        return 'Rendah'
    elif prob_diabetes < 0.6:
        return 'Sedang'
    else:
        return 'Tinggi'

def get_recommendation(prediction, prob_diabetes, features):
    """
    Memberikan rekomendasi berdasarkan hasil prediksi dan profil pasien
    """
    glucose = features[1]
    bmi = features[5]
    age = features[7]
    
    recommendations = []
    
    if prediction == 1 or prob_diabetes > 0.5:
        recommendations.append("⚠️ Segera konsultasi dengan dokter untuk pemeriksaan lebih lanjut.")
        recommendations.append("🩸 Lakukan tes HbA1c dan gula darah puasa.")
    else:
        recommendations.append("✅ Kondisi saat ini menunjukkan risiko rendah diabetes.")
    
    if glucose > 140:
        recommendations.append("📊 Kadar glukosa Anda tinggi. Monitor gula darah secara rutin.")
    
    if bmi > 30:
        recommendations.append("⚖️ BMI Anda menunjukkan obesitas. Pertimbangkan program penurunan berat badan.")
    elif bmi > 25:
        recommendations.append("⚖️ BMI Anda dalam kategori overweight. Jaga pola makan dan olahraga teratur.")
    
    if age > 45:
        recommendations.append("👴 Usia Anda termasuk faktor risiko. Lakukan screening rutin setiap tahun.")
    
    recommendations.append("🏃 Olahraga minimal 30 menit setiap hari.")
    recommendations.append("🥗 Konsumsi makanan sehat dengan gizi seimbang.")
    
    return recommendations

@app.route('/info')
def info():
    """
    Route untuk informasi tentang model
    """
    info_data = {
        'model': 'K-Nearest Neighbors (KNN)',
        'k_value': 8,
        'accuracy': '76.62%',
        'precision': '71.43%',
        'recall': '55.56%',
        'f1_score': '62.50%',
        'dataset': 'Pima Indians Diabetes Database',
        'total_samples': 768,
        'features': feature_names
    }
    return jsonify(info_data)

# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("APLIKASI PREDIKSI DIABETES - K-NEAREST NEIGHBORS (KNN)")
    print("Putra Aliansyah (301230041) - IF 5A")
    print("="*80)
    
    # Load atau train model
    if load_model():
        print("\n[INFO] Server siap dijalankan!")
        print("[INFO] Akses aplikasi di: http://localhost:5000")
        print("[INFO] Tekan CTRL+C untuk stop server\n")
        
        # Jalankan Flask server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n[ERROR] Gagal memuat model. Pastikan file diabetes.csv tersedia.")
        print("[ERROR] Download dataset dari: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")