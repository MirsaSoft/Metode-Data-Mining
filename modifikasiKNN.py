import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

class ModifiedKNN:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.best_k = None
        self.best_model = None
        
    def preprocess_data(self, df):
        # Menghapus kolom yang tidak diperlukan
        columns_to_drop = ['ID_Nasabah', 'Nama_Nasabah']
        df = df.drop(columns_to_drop, axis=1)
        
        # Encoding kategorikal
        categorical_columns = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 
                             'Pekerjaan', 'Riwayat_Kredit', 'Status_Rumah', 'Status_Pinjaman']
        df = pd.get_dummies(df, columns=categorical_columns)
        
        return df
    
    def find_optimal_k(self, X_train, y_train):
        k_range = range(1, 31, 2)
        scores = []
        
        print("\n" + "="*50)
        print("PENCARIAN NILAI K OPTIMAL")
        print("="*50)
        print("\nMencari nilai k optimal dari range 1-30...")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            score = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            scores.append(np.mean(score))
            print(f"K = {k:2d} | Akurasi CV = {np.mean(score)*100:.2f}%")
        
        self.best_k = k_range[np.argmax(scores)]
        print(f"\nNilai K optimal yang ditemukan: {self.best_k}")
        print(f"Akurasi tertinggi: {max(scores)*100:.2f}%")
        
        # Visualisasi pencarian k optimal
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, marker='o')
        plt.xlabel('Nilai K')
        plt.ylabel('Akurasi Cross-validation')
        plt.title('Pencarian Nilai K Optimal')
        plt.grid(True)
        plt.show()
        
        return self.best_k
    
    def train(self, X, y):
        print("\n" + "="*50)
        print("PROSES TRAINING DAN EVALUASI MODEL")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nJumlah data training: {len(X_train)}")
        print(f"Jumlah data testing: {len(X_test)}")
        
        # Scaling fitur
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Mencari k optimal
        optimal_k = self.find_optimal_k(X_train_scaled, y_train)
        
        # Membuat ensemble KNN dengan berbagai weights
        knn1 = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform')
        knn2 = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
        
        # Voting classifier
        self.best_model = VotingClassifier(
            estimators=[('uniform', knn1), ('distance', knn2)],
            voting='soft'
        )
        
        # Training
        print("\nMelatih model ensemble KNN...")
        self.best_model.fit(X_train_scaled, y_train)
        
        # Prediksi
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("HASIL EVALUASI MODEL")
        print("="*50)
        
        # Menampilkan classification report dalam format persentase
        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nLaporan Klasifikasi Detail:")
        print("-"*50)
        
        # Mengubah format laporan menjadi persentase
        for label in report.keys():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"\nKelas: {label}")
                print(f"Precision: {report[label]['precision']*100:.2f}%")
                print(f"Recall: {report[label]['recall']*100:.2f}%")
                print(f"F1-score: {report[label]['f1-score']*100:.2f}%")
                print(f"Support: {report[label]['support']}")
        
        print("\n" + "-"*50)
        print(f"Akurasi Overall: {accuracy*100:.2f}%")
        print("-"*50)
        
        # Visualisasi confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.show()
        
        return accuracy * 100
    
    def predict(self, X_new):
        X_new_scaled = self.scaler.transform(X_new)
        return self.best_model.predict(X_new_scaled)

# Penggunaan
def main():
    print("\n" + "="*50)
    print("SISTEM PREDIKSI PINJAMAN MENGGUNAKAN MODIFIED KNN")
    print("="*50)
    
    # Baca dataset
    print("\nMembaca dataset...")
    df = pd.read_excel("C:\Kuliah\Python\dataset_pinjaman_nasabah_formatted (600).xlsx")
    print(f"Jumlah total data: {len(df)}")
    
    # Inisialisasi model
    model = ModifiedKNN()
    
    # Preprocessing
    print("\nMemproses data...")
    processed_df = model.preprocess_data(df)
    
    # Pisahkan fitur dan target
    X = processed_df.drop('Status_Pinjaman_Diterima', axis=1)
    y = processed_df['Status_Pinjaman_Diterima']
    
    # Training dan evaluasi
    accuracy = model.train(X, y)

if __name__ == "__main__":
    main()