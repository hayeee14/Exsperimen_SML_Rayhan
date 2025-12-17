import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os

# Library Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- KONFIGURASI ---
EXPERIMENT_NAME = "Eksperimen_Loan_Rayhan_Skilled"
ARTIFACT_PATH = "model_random_forest_tuned"

def run_training():
    # --- 1. LOAD DATA ---
    # Mencari file data bersih
    file_path = os.path.join("namadataset_preprocessing", "train_clean.csv")
    
    print(f"📂 Sedang membaca data dari: {file_path}")
    
    # Cek apakah file ada
    if not os.path.exists(file_path):
        print("❌ ERROR: File 'train_clean.csv' tidak ditemukan!")
        print("👉 Pastikan file tersebut ada di dalam folder 'Membangun_model/namadataset_preprocessing'")
        return

    df = pd.read_csv(file_path)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. SETUP MLFLOW ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Skilled_Hyperparameter_Tuning"):
        print("🚀 Memulai Training dengan Hyperparameter Tuning...")

        # --- 3. TUNING ---
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"✅ Parameter Terbaik: {best_params}")

        # --- 4. EVALUASI ---
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        print(f"📊 Akurasi: {acc:.4f}")

        # --- 5. LOGGING MANUAL (Syarat Skilled) ---
        print("📝 Mencatat ke MLflow...")
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        mlflow.sklearn.log_model(best_model, ARTIFACT_PATH)
        print("\n🎉 SUKSES! Silakan cek MLflow UI.")

if __name__ == "__main__":
    run_training()