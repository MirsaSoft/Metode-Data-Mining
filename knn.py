import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk membaca dan mempersiapkan data
def load_and_prepare_data(file_path):
    # Membaca dataset
    df = pd.read_excel(r"C:\Users\sdasu\Desktop\testing\dataset_pinjaman_nasabah_formatted (600).xlsx")
    
    # Menghapus kolom yang tidak diperlukan
    columns_to_drop = ['ID_Nasabah', 'Nama_Nasabah']
    df = df.drop(columns=columns_to_drop)
    
    # Mengubah variable kategorikal menjadi numerical
    categorical_columns = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 
                         'Pekerjaan', 'Riwayat_Kredit', 'Status_Rumah', 'Status_Pinjaman']
    
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    
    return df

# Fungsi untuk membagi dataset dan melakukan scaling
def preprocess_data(df):
    X = df.drop('Status_Pinjaman', axis=1)
    y = df['Status_Pinjaman']
    
    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Melakukan scaling pada fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Fungsi untuk mencari k optimal
def find_optimal_k(X_train, X_test, y_train, y_test, k_range):
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
    
    optimal_k = k_range[np.argmax(accuracies)]
    return optimal_k, accuracies

# Fungsi untuk visualisasi hasil
def plot_results(k_range, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel('Nilai K')
    plt.ylabel('Akurasi')
    plt.title('Akurasi KNN berdasarkan Nilai K')
    plt.grid(True)
    plt.show()

# Fungsi utama
def main():
    # Load data
    file_path = "dataset_pinjaman_nasabah_formatted (600).xlsx"
    df = load_and_prepare_data(file_path)
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    # Mencari k optimal
    k_range = range(1, 31, 2)  # Mencoba k ganjil dari 1 sampai 30
    optimal_k, accuracies = find_optimal_k(X_train_scaled, X_test_scaled, y_train, y_test, k_range)
    
    # Melatih model dengan k optimal
    final_model = KNeighborsClassifier(n_neighbors=optimal_k)
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluasi model
    y_pred = final_model.predict(X_test_scaled)
    
    # Menampilkan hasil
    print(f"\nNilai K optimal: {optimal_k}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAkurasi: {accuracy_score(y_test, y_pred):.4f}")
    
    # Visualisasi hasil
    plot_results(k_range, accuracies)

if __name__ == "__main__":
    main()