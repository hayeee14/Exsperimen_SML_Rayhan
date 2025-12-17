import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- KONFIGURASI ---
EXPERIMENT_NAME = "Eksperimen_Loan_Basic"

def run_basic_training():
    # 1. Load Data
    # Pastikan lokasi file sesuai dengan struktur folder kamu
    file_path = os.path.join("namadataset_preprocessing", "train_clean.csv")
    
    if not os.path.exists(file_path):
        print("❌ File train_clean.csv tidak ditemukan!")
        return

    df = pd.read_csv(file_path)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # --- FITUR AUTOLOG (Kunci Syarat Basic) ---
    # Perintah ini akan otomatis mencatat semua parameter & metrics tanpa kita ketik manual
    mlflow.autolog()

    with mlflow.start_run(run_name="Basic_Training_Autolog"):
        print("🚀 Memulai Basic Training (Mode Autolog)...")
        
        # Training Model Sederhana (Tanpa Tuning)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        # Cukup hitung score, MLflow akan mencatat sisanya otomatis
        acc = model.score(X_test, y_test)
        
        print(f"✅ Training Selesai. Akurasi: {acc:.4f}")
        print("Catatan: Parameter dan metrics sudah direkam otomatis oleh MLflow.")

if __name__ == "__main__":
    run_basic_training()