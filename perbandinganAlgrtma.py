import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class LoanPredictionComparison:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.nb_model = GaussianNB()
        self.knn_model = None
        self.svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        self.feature_names = None
        
    def load_data(self, file_path):
        print("\n" + "="*50)
        print("SISTEM PERBANDINGAN PREDIKSI PINJAMAN")
        print("="*50)
        
        print("\nMembaca dataset...")
        try:
            df = pd.read_excel(file_path)
            print(f"Jumlah total data: {len(df)}")
            return df
        except Exception as e:
            print(f"Error membaca file: {str(e)}")
            return None
    
    def prepare_data(self, df):
        print("\nMempersiapkan data...")
        # menghapus fitur yang tidak releval dengan perhitungan
        columns_to_drop = ['ID_Nasabah', 'Nama_Nasabah'] if 'ID_Nasabah' in df.columns else []
        df = df.drop(columns_to_drop, axis=1)
        
        
        self.feature_names = df.columns.tolist()
        self.feature_names.remove('Status_Pinjaman')
        
        
        categorical_columns = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 
                             'Pekerjaan', 'Riwayat_Kredit', 'Status_Rumah', 'Status_Pinjaman']
        
        for column in categorical_columns:
            if column in df.columns:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
        return df
    
    def preprocess_features(self, X):
        numerical_columns = ['Usia', 'Jumlah_Tanggungan', 'Lama_Bekerja', 
                           'Pendapatan_Perbulan', 'Pendapatan_Tambahan',
                           'Jumlah_Pinjaman', 'Jangka_Waktu', 'Lama_Tinggal',
                           'Rasio_Pinjaman_Pendapatan']
        
        X_scaled = X.copy()
        numerical_cols = [col for col in numerical_columns if col in X.columns]
        X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        return X_scaled
    
    def find_optimal_k(self, X_train, X_test, y_train, y_test, k_range):
        accuracies = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            accuracy = knn.score(X_test, y_test)
            accuracies.append(accuracy)
        
        optimal_k = k_range[np.argmax(accuracies)]
        return optimal_k, accuracies
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Train and evaluate Naive Bayes
        print("\nTraining Naive Bayes...")
        self.nb_model.fit(X_train, y_train)
        nb_pred = self.nb_model.predict(X_test)
        nb_prob = self.nb_model.predict_proba(X_test)[:, 1]

        
        # Train and evaluate SVM
        print("Training SVM...")
        self.svm_model.fit(X_train, y_train)
        svm_pred = self.svm_model.predict(X_test)
        
        # Find optimal k for KNN
        print("Training KNN dan mencari k optimal...")
        k_range = range(1, 31, 2)
        optimal_k, knn_accuracies = self.find_optimal_k(X_train, X_test, y_train, y_test, k_range)
        
        # Train and evaluate KNN with optimal k
        self.knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
        self.knn_model.fit(X_train, y_train)
        knn_pred = self.knn_model.predict(X_test)
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        
        return {
            'nb': {
                'predictions': nb_pred,
                'probabilities': nb_prob,
                'accuracy': accuracy_score(y_test, nb_pred),
                'conf_matrix': confusion_matrix(y_test, nb_pred),
                'class_report': classification_report(y_test, nb_pred, output_dict=True)
            },
            'knn': {
                'predictions': knn_pred,
                'accuracy': accuracy_score(y_test, knn_pred),
                'conf_matrix': confusion_matrix(y_test, knn_pred),
                'class_report': classification_report(y_test, knn_pred, output_dict=True),
                'optimal_k': optimal_k,
                'k_accuracies': knn_accuracies
            },
            'svm': {
                'predictions': svm_pred,
                'accuracy': accuracy_score(y_test, svm_pred),
                'conf_matrix': confusion_matrix(y_test, svm_pred),
                'class_report': classification_report(y_test, svm_pred, output_dict=True)
            }
        }
    
    def plot_comparison_results(self, results, k_range):
        # Plot confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = ['nb', 'knn', 'svm']
        titles = ['Naive Bayes', 'KNN', 'SVM']
        
        for i, (model, title) in enumerate(zip(models, titles)):
            sns.heatmap(results[model]['conf_matrix'], 
                       annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {title}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        algorithms = ['Naive Bayes', 'KNN', 'SVM']
        accuracies = [results['nb']['accuracy'], 
                     results['knn']['accuracy'],
                     results['svm']['accuracy']]
        
        bars = plt.bar(algorithms, [acc * 100 for acc in accuracies])
        plt.title('Perbandingan Akurasi Model')
        plt.ylabel('Akurasi (%)')
        
        # Add accuracy values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Plot KNN k-value analysis
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, results['knn']['k_accuracies'], marker='o')
        plt.xlabel('Nilai K')
        plt.ylabel('Akurasi')
        plt.title(f'Analisis Nilai K (Optimal K = {results["knn"]["optimal_k"]})')
        plt.grid(True)
        plt.show()

def main():
    system = LoanPredictionComparison()
    
    try:
        # Load and prepare data
        file_path = r"C:\Users\sdasu\Desktop\testing\dataset_pinjaman_nasabah_formatted (600).xlsx"  # Sesuaikan dengan path file Anda
        df = system.load_data(file_path)
        
        
        if df is None:
            return
        
        df = system.prepare_data(df)
        
        # Split features and target
        X = df.drop('Status_Pinjaman', axis=1)
        y = df['Status_Pinjaman']
        
        # Preprocess features
        X_processed = system.preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42 
        )
        
        # Train and evaluate models
        results = system.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Display results
        print("\n" + "="*50)
        print("HASIL PERBANDINGAN MODEL")
        print("="*50)
        
        for model_name in ['nb', 'knn', 'svm']:
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {results[model_name]['accuracy']*100:.2f}%")
            print("\nClassification Report:")
            report = results[model_name]['class_report']
            
            for label in report.keys():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"\nKelas: {label}")
                    print(f"Precision: {report[label]['precision']*100:.2f}%")
                    print(f"Recall: {report[label]['recall']*100:.2f}%")
                    print(f"F1-score: {report[label]['f1-score']*100:.2f}%")
                    print(f"Support: {report[label]['support']}")
        
        if 'optimal_k' in results['knn']:
            print(f"\nKNN Optimal K: {results['knn']['optimal_k']}")
        
        # Plot results
        system.plot_comparison_results(results, range(1, 31, 2))
        
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        raise

if __name__ == "__main__":
    main()