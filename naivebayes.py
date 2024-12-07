import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

class LoanPredictionSystem:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = GaussianNB()
        self.feature_names = None
        
    def prepare_data(self, file_path):
        """
        Membaca dan mempersiapkan dataset untuk pemrosesan
        """
        # Membaca dataset
        try:
            df = pd.read_excel(r"C:\Users\sdasu\Desktop\testing\dataset_pinjaman_nasabah_formatted (600).xlsx")
        except Exception as e:
            print(f"Error membaca file: {str(e)}")
            return None
            
        # Menghapus kolom yang tidak diperlukan
        columns_to_drop = ['ID_Nasabah', 'Nama_Nasabah']
        df = df.drop(columns_to_drop, axis=1)
        
        # Menyimpan nama-nama feature
        self.feature_names = df.columns.tolist()
        self.feature_names.remove('Status_Pinjaman')
        
        # Mengubah categorical features menjadi numerical
        categorical_columns = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 
                             'Pekerjaan', 'Riwayat_Kredit', 'Status_Rumah', 'Status_Pinjaman']
        
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        return df
    
    def handle_imbalanced_data(self, X, y):
        """
        Menangani imbalanced data menggunakan SMOTE
        """
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def preprocess_features(self, X):
        """
        Melakukan preprocessing pada features
        """
        # Normalisasi numerical features
        numerical_columns = ['Usia', 'Jumlah_Tanggungan', 'Lama_Bekerja', 
                           'Pendapatan_Perbulan', 'Pendapatan_Tambahan',
                           'Jumlah_Pinjaman', 'Jangka_Waktu', 'Lama_Tinggal',
                           'Rasio_Pinjaman_Pendapatan']
        
        X_scaled = X.copy()
        X_scaled[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        return X_scaled
    
    def train_model(self, X_train, y_train):
        """
        Melatih model Naive Bayes
        """
        self.model.fit(X_train, y_train)
        
    def evaluate_model(self, X_test, y_test):
        """
        Mengevaluasi performa model
        """
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Menghitung ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_data': (fpr, tpr, roc_auc)
        }
    
    def plot_results(self, evaluation_results):
        """
        Memvisualisasikan hasil evaluasi model
        """
        # Plot confusion matrix
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot ROC curve
        plt.subplot(1, 2, 2)
        fpr, tpr, roc_auc = evaluation_results['roc_data']
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, new_data):
        """
        Melakukan prediksi untuk data baru
        """
        # Preprocess data baru
        for column, encoder in self.label_encoders.items():
            if column in new_data.columns and column != 'Status_Pinjaman':
                new_data[column] = encoder.transform(new_data[column])
        
        # Normalisasi numerical features
        new_data_scaled = self.preprocess_features(new_data)
        
        # Prediksi
        prediction = self.model.predict(new_data_scaled)
        prediction_prob = self.model.predict_proba(new_data_scaled)
        
        return prediction, prediction_prob

def main():
    # Inisialisasi sistem
    loan_system = LoanPredictionSystem()
    
    try:
        # Persiapkan data
        file_path = "dataset_pinjaman_nasabah_formatted (600).xlsx"
        df = loan_system.prepare_data(file_path)
        
        if df is None:
            return
        
        # Split features dan target
        X = df.drop('Status_Pinjaman', axis=1)
        y = df['Status_Pinjaman']
        
        # Preprocessing features
        X_processed = loan_system.preprocess_features(X)
        
        # Handle imbalanced data
        X_balanced, y_balanced = loan_system.handle_imbalanced_data(X_processed, y)
        
        # Split data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        
        # Train model
        loan_system.train_model(X_train, y_train)
        
        # Evaluasi model
        evaluation_results = loan_system.evaluate_model(X_test, y_test)
        
        # Tampilkan hasil
        print("\nHasil Evaluasi Model Naive Bayes:")
        print("-" * 50)
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(evaluation_results['confusion_matrix'])
        print("\nClassification Report:")
        print(evaluation_results['classification_report'])
        
        # Visualisasi hasil
        loan_system.plot_results(evaluation_results)
        
        # Cross-validation
        cv_scores = cross_val_score(loan_system.model, X_balanced, y_balanced, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        
    return loan_system

if __name__ == "__main__":
    loan_system = main()