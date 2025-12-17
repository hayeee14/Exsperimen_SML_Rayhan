import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def perform_preprocessing(df):
    """
    Fungsi ini menerima DataFrame mentah (df) dan mengembalikan 
    X (fitur) dan y (target) yang sudah bersih, di-encoding, dan di-scaling.
    """
    
    df_clean = df.copy()

    
    
    
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in categorical_cols:
        if col in df_clean.columns:
        
            modus = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(modus)

    
    numerical_cols_na = ['LoanAmount', 'Loan_Amount_Term']
    for col in numerical_cols_na:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # --- CLEANING DATA ---
    
    
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop('Loan_ID', axis=1)

    
    if 'Dependents' in df_clean.columns:
        df_clean['Dependents'] = df_clean['Dependents'].replace('3+', '3')

    

    
    y = None
    X_raw = df_clean 

    
    if 'Loan_Status' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Loan_Status'] = le.fit_transform(df_clean['Loan_Status'])
        y = df_clean['Loan_Status']
        # Pisahkan fitur (X) dari target (y)
        X_raw = df_clean.drop('Loan_Status', axis=1)

    # One-Hot Encoding untuk Fitur Kategorikal (Gender, Married, Education, dll)
    # drop_first=True untuk mencegah dummy variable trap
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    # --- SCALING (NORMALISASI) ---
    
    scaler = StandardScaler()
    # Hasil fit_transform adalah array numpy, ubah kembali ke DataFrame
    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

    return X_scaled, y

# --- BLOK UTAMA (MAIN PROGRAM) ---
if __name__ == "__main__":
    # --- 1. KONFIGURASI LOKASI ---
    folder_dataset = 'LoanPrediction_raw'
    input_file = 'train.csv'
    output_file_name = 'train_clean.csv'  # Nama file hasil preprocessing
    
    # Mendapatkan lokasi script ini
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Menyusun jalur file input (Data Mentah)
    path_input = os.path.join(base_dir, '..', folder_dataset, input_file)
    # Menyusun jalur file output (Data Bersih)
    path_output = os.path.join(base_dir, '..', folder_dataset, output_file_name)

    print(f" Mencari dataset mentah di: {os.path.abspath(path_input)}")

    try:
        # --- 2. PROSES LOAD & PREPROCESSING ---
        df_mentah = pd.read_csv(path_input)
        print(" Dataset ditemukan! Memulai preprocessing...")
        
        # Panggil fungsi otomatisasi
        X_hasil, y_hasil = perform_preprocessing(df_mentah)
        
        # --- 3. PENYIMPANAN DATA (SAVE TO CSV) ---
        print("\n Menyimpan hasil preprocessing...")
        
        # Kita gabungkan kembali X (Fitur) dan y (Target) agar jadi satu file CSV utuh
        if y_hasil is not None:
            # Ubah y (array) menjadi Series agar punya nama kolom
            y_series = pd.Series(y_hasil, name='Loan_Status')
            # Gabungkan berdampingan (axis=1)
            df_lengkap = pd.concat([X_hasil, y_series], axis=1)
        else:
            df_lengkap = X_hasil
        
        # Simpan ke CSV tanpa index angka (0,1,2..)
        df_lengkap.to_csv(path_output, index=False)
        
        # --- 4. LAPORAN SUKSES ---
        print("\n" + "="*40)
        print("SUKSES! DATASET BERSIH TELAH DIBUAT")
        print("="*40)
        print(f"Lokasi File: {os.path.abspath(path_output)}")
        print(f"Ukuran Data: {df_lengkap.shape} (Baris, Kolom)")
        print("\nCek folder 'LoanPrediction_raw', file 'train_clean.csv' sudah ada di sana.")
        
    except FileNotFoundError:
        print(f"\n❌ [ERROR] File '{input_file}' tidak ditemukan.")
        print("Pastikan struktur folder benar dan nama folder 'LoanPrediction_raw' tidak salah ketik.")
    except Exception as e:
        print(f"\n❌ [ERROR] Terjadi kesalahan: {e}")