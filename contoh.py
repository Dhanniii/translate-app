"""
LANGUAGE DETECTION - NAIVE BAYES
Kelompok 5 - Project Capstone Bu Erna
Bidang NLP: Text Classification

Program ini mengklasifikasikan teks ke dalam 2 bahasa:
- Indonesian
- English

Menggunakan dataset dari dataset.json yang sudah ada.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator

# ============================================================================
# FUNGSI 1: LOAD DATASET
# ============================================================================

def load_dataset(file_path):
    """
    Membaca dataset dari file JSON dan convert ke format klasifikasi
    
    Parameter:
        file_path (str): Path ke file dataset.json
    
    Return:
        DataFrame: Data dengan kolom 'text' dan 'label'
    """
    print("=" * 80)
    print("📂 LOADING DATASET...")
    print("=" * 80)
    
    # Baca file JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ambil data dari JSON
    dataset = data['data']
    
    # Buat list untuk menyimpan text dan label
    texts = []
    labels = []
    
    # Loop setiap data
    for item in dataset:
        # Tambahkan teks Indonesia dengan label "Indonesian"
        texts.append(item['indonesia'])
        labels.append('Indonesian')
        
        # Tambahkan teks English dengan label "English"
        texts.append(item['english'])
        labels.append('English')
    
    # Buat DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"✅ Dataset berhasil dimuat!")
    print(f"📊 Total data: {len(df)}")
    print(f"   - Indonesian: {len(df[df['label'] == 'Indonesian'])}")
    print(f"   - English: {len(df[df['label'] == 'English'])}")
    print()
    
    return df


# ============================================================================
# FUNGSI 1B: LOAD DATASET DARI INPUT USER
# ============================================================================

def load_dataset_from_input():
    """
    Membuat dataset dari input user dengan translate otomatis
    
    Return:
        DataFrame: Data dengan kolom 'text' dan 'label'
    """
    print("=" * 80)
    print("📝 INPUT DATASET MANUAL")
    print("=" * 80)
    print("Masukkan kalimat dalam bahasa Indonesia (ketik 'selesai' untuk berhenti)")
    print()
    
    texts = []
    labels = []
    translator = GoogleTranslator(source='id', target='en')
    
    count = 1
    while True:
        print(f"\n--- Data ke-{count} ---")
        text_id = input("Kalimat Indonesia: ")
        
        if text_id.lower() == 'selesai':
            break
        
        if text_id.strip() == "":
            print("❌ Teks tidak boleh kosong!")
            continue
        
        # Translate otomatis
        print("🔄 Menerjemahkan...")
        try:
            text_en = translator.translate(text_id)
            print(f"✅ Hasil translate: {text_en}")
            
            # Tambahkan ke dataset
            texts.append(text_id)
            labels.append('Indonesian')
            texts.append(text_en)
            labels.append('English')
            
            count += 1
        except Exception as e:
            print(f"❌ Error translate: {e}")
            continue
    
    # Buat DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print("\n" + "=" * 80)
    print(f"✅ Dataset berhasil dibuat!")
    print(f"📊 Total data: {len(df)}")
    print(f"   - Indonesian: {len(df[df['label'] == 'Indonesian'])}")
    print(f"   - English: {len(df[df['label'] == 'English'])}")
    print("=" * 80)
    print()
    
    return df


# ============================================================================
# FUNGSI 1C: PILIH SUMBER DATASET
# ============================================================================

def choose_dataset_source():
    """
    Memilih sumber dataset: dari file JSON atau input manual
    
    Return:
        DataFrame: Data dengan kolom 'text' dan 'label'
    """
    print("\n")
    print("=" * 80)
    print("           🌐 LANGUAGE DETECTION - NAIVE BAYES")
    print("           Kelompok 5 - Project Capstone Bu Erna")
    print("=" * 80)
    print()
    print("PILIH SUMBER DATASET:")
    print("1. Load dari file dataset.json (50 pasangan kalimat)")
    print("2. Input manual dengan translate otomatis")
    print()
    
    while True:
        pilihan = input("Pilih opsi (1/2): ")
        
        if pilihan == '1':
            return load_dataset('dataset.json')
        elif pilihan == '2':
            df = load_dataset_from_input()
            if len(df) < 10:
                print("\n⚠️  WARNING: Dataset terlalu sedikit (minimal 10 pasangan kalimat)")
                print("Untuk hasil yang baik, masukkan minimal 10 kalimat.\n")
                ulang = input("Mau input lagi? (y/n): ")
                if ulang.lower() == 'y':
                    continue
            return df
        else:
            print("❌ Pilihan tidak valid! Pilih 1 atau 2.\n")


# ============================================================================
# FUNGSI 2: PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """
    Preprocessing teks sederhana
    
    Parameter:
        text (str): Teks yang akan diproses
    
    Return:
        str: Teks yang sudah diproses
    """
    # Lowercase
    text = text.lower()
    
    # Hapus tanda baca (opsional, bisa di-skip untuk language detection)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text


# ============================================================================
# FUNGSI 3: SPLIT DATA
# ============================================================================

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data menjadi training dan testing
    
    Parameter:
        df (DataFrame): Dataset
        test_size (float): Persentase data testing (default 0.2 = 20%)
        random_state (int): Random seed untuk reproducibility
    
    Return:
        tuple: X_train, X_test, y_train, y_test
    """
    print("=" * 80)
    print("✂️  SPLITTING DATA...")
    print("=" * 80)
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Pastikan proporsi label seimbang
    )
    
    print(f"📊 Data Training: {len(X_train)} samples")
    print(f"📊 Data Testing: {len(X_test)} samples")
    print()
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# FUNGSI 4: VECTORIZATION
# ============================================================================

def vectorize_text(X_train, X_test):
    """
    Convert teks menjadi vektor numerik menggunakan TF-IDF
    
    Parameter:
        X_train: Data training
        X_test: Data testing
    
    Return:
        tuple: X_train_vec, X_test_vec, vectorizer
    """
    print("=" * 80)
    print("🔢 VECTORIZATION (TF-IDF)...")
    print("=" * 80)
    
    # Inisialisasi TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Ambil 1000 kata paling penting
        ngram_range=(1, 2),  # Unigram dan bigram
        min_df=2  # Kata harus muncul minimal 2x
    )
    
    # Fit dan transform data training
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Transform data testing (pakai vectorizer yang sama)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Vectorization selesai!")
    print(f"📊 Feature (kata unik): {len(vectorizer.get_feature_names_out())}")
    print()
    
    return X_train_vec, X_test_vec, vectorizer


# ============================================================================
# FUNGSI 5: TRAINING MODEL
# ============================================================================

def train_model(X_train_vec, y_train):
    """
    Training model Naive Bayes
    
    Parameter:
        X_train_vec: Data training yang sudah di-vectorize
        y_train: Label training
    
    Return:
        model: Model yang sudah di-train
    """
    print("=" * 80)
    print("🤖 TRAINING MODEL (NAIVE BAYES)...")
    print("=" * 80)
    
    # Inisialisasi Multinomial Naive Bayes
    model = MultinomialNB()
    
    # Training
    model.fit(X_train_vec, y_train)
    
    print("✅ Model berhasil di-train!")
    print()
    
    return model


# ============================================================================
# FUNGSI 6: EVALUASI MODEL
# ============================================================================

def evaluate_model(model, X_test_vec, y_test):
    """
    Evaluasi model dengan Classification Report dan Confusion Matrix
    
    Parameter:
        model: Model yang sudah di-train
        X_test_vec: Data testing yang sudah di-vectorize
        y_test: Label testing
    
    Return:
        y_pred: Prediksi model
    """
    print("=" * 80)
    print("📊 EVALUASI MODEL")
    print("=" * 80)
    
    # Prediksi
    y_pred = model.predict(X_test_vec)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Classification Report
    print("=" * 80)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("=" * 80)
    print("📊 CONFUSION MATRIX")
    print("=" * 80)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    
    return y_pred, cm


# ============================================================================
# FUNGSI 7: VISUALISASI CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(cm, labels=['English', 'Indonesian']):
    """
    Visualisasi Confusion Matrix dengan heatmap
    
    Parameter:
        cm: Confusion matrix
        labels: Label kelas
    """
    print("=" * 80)
    print("📈 VISUALISASI CONFUSION MATRIX")
    print("=" * 80)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Language Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Simpan gambar
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion Matrix disimpan sebagai 'confusion_matrix.png'")
    
    plt.show()
    print()


# ============================================================================
# FUNGSI 8: TESTING PREDIKSI
# ============================================================================

def predict_language(text, model, vectorizer):
    """
    Prediksi bahasa dari teks baru
    
    Parameter:
        text (str): Teks yang akan diprediksi
        model: Model yang sudah di-train
        vectorizer: Vectorizer yang sudah di-fit
    
    Return:
        str: Prediksi bahasa (Indonesian/English)
    """
    # Preprocess
    text_processed = preprocess_text(text)
    
    # Vectorize
    text_vec = vectorizer.transform([text_processed])
    
    # Prediksi
    prediction = model.predict(text_vec)[0]
    
    # Probabilitas
    proba = model.predict_proba(text_vec)[0]
    
    return prediction, proba


# ============================================================================
# FUNGSI 9: TESTING INTERAKTIF
# ============================================================================

def interactive_testing(model, vectorizer):
    """
    Testing interaktif untuk prediksi teks baru
    
    Parameter:
        model: Model yang sudah di-train
        vectorizer: Vectorizer yang sudah di-fit
    """
    print("=" * 80)
    print("🧪 TESTING - PREDIKSI TEKS BARU")
    print("=" * 80)
    print("Ketik teks untuk diprediksi bahasanya (atau 'exit' untuk keluar)")
    print()
    
    while True:
        text = input("📝 Masukkan teks: ")
        
        if text.lower() == 'exit':
            print("\n✅ Terima kasih!")
            break
        
        if text.strip() == "":
            print("❌ Teks tidak boleh kosong!\n")
            continue
        
        # Prediksi
        prediction, proba = predict_language(text, model, vectorizer)
        
        # Tampilkan hasil
        print("\n" + "-" * 80)
        print(f"📄 Teks: {text}")
        print(f"🌍 Prediksi: {prediction}")
        print(f"📊 Probabilitas:")
        print(f"   - English: {proba[0]:.4f} ({proba[0]*100:.2f}%)")
        print(f"   - Indonesian: {proba[1]:.4f} ({proba[1]*100:.2f}%)")
        print("-" * 80)
        print()


# ============================================================================
# FUNGSI MAIN
# ============================================================================

def main():
    """
    Fungsi utama program
    """
    # 1. Pilih Sumber Dataset (dari file atau input manual)
    df = choose_dataset_source()
    
    # 2. Preprocessing (sudah dilakukan di dalam fungsi)
    df['text'] = df['text'].apply(preprocess_text)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 4. Vectorization
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # 5. Training
    model = train_model(X_train_vec, y_train)
    
    # 6. Evaluasi
    y_pred, cm = evaluate_model(model, X_test_vec, y_test)
    
    # 7. Visualisasi Confusion Matrix
    plot_confusion_matrix(cm)
    
    # 8. Testing Interaktif
    interactive_testing(model, vectorizer)


# ============================================================================
# JALANKAN PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()
