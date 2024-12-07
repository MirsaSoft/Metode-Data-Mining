import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data
def load_data(file_path):
    # Read Excel file
    df = pd.read_excel(r"C:\Users\sdasu\Desktop\testing\dataset_pinjaman_nasabah_formatted (600).xlsx")
    return df

# Function to prepare features and target
def prepare_data(df):
    # Convert categorical variables to numeric using Label Encoding
    le = LabelEncoder()
    categorical_columns = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 
                         'Pekerjaan', 'Riwayat_Kredit', 'Status_Rumah']
    
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    # Select features
    features = ['Jenis_Kelamin', 'Usia', 'Status_Pernikahan', 'Jumlah_Tanggungan',
               'Pendidikan', 'Pekerjaan', 'Lama_Bekerja', 'Pendapatan_Perbulan',
               'Pendapatan_Tambahan', 'Riwayat_Kredit', 'Status_Rumah', 
               'Lama_Tinggal', 'Rasio_Pinjaman_Pendapatan']
    
    X = df[features]
    y = df['Status_Pinjaman']
    
    return X, y

# Function to train SVM model
def train_svm_model(X_train, y_train):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize and train SVM model
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    return svm, scaler

# Function to evaluate model
def evaluate_model(model, X_test, y_test, scaler):
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm, y_pred

# Function to visualize results
def visualize_results(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load data
    file_path = "dataset_pinjaman_nasabah_formatted (600).xlsx"
    df = load_data(file_path)
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    svm_model, scaler = train_svm_model(X_train, y_train)
    
    # Evaluate model
    accuracy, report, cm, y_pred = evaluate_model(svm_model, X_test, y_test, scaler)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize results
    visualize_results(cm)
    
    return svm_model, scaler

if __name__ == "__main__":
    model, scaler = main()