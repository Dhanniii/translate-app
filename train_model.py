"""
TRAIN AND SAVE MODEL
Script untuk training model dan menyimpannya sebagai pickle file

Usage:
    python train_model.py
"""

import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD DATASET
# ============================================================================

def load_dataset(file_path='dataset.json'):
    """Load dataset from JSON file"""
    print("=" * 80)
    print("📂 LOADING DATASET...")
    print("=" * 80)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data['data']
    
    texts = []
    labels = []
    
    for item in dataset:
        texts.append(item['indonesia'])
        labels.append('Indonesian')
        texts.append(item['english'])
        labels.append('English')
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"   - Indonesian: {len(df[df['label'] == 'Indonesian'])}")
    print(f"   - English: {len(df[df['label'] == 'English'])}")
    print()
    
    return df

# ============================================================================
# PREPROCESS
# ============================================================================

def preprocess_text(text):
    """Simple preprocessing"""
    return text.lower()

# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_and_save_model():
    """Train model and save to pickle files"""
    
    # 1. Load dataset
    df = load_dataset()
    
    # 2. Preprocess
    df['text'] = df['text'].apply(preprocess_text)
    
    # 3. Split data
    print("=" * 80)
    print("✂️  SPLITTING DATA...")
    print("=" * 80)
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"📊 Training data: {len(X_train)} samples")
    print(f"📊 Testing data: {len(X_test)} samples")
    print()
    
    # 4. Vectorization
    print("=" * 80)
    print("🔢 VECTORIZATION (TF-IDF)...")
    print("=" * 80)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Features: {len(vectorizer.get_feature_names_out())}")
    print()
    
    # 5. Train model
    print("=" * 80)
    print("🤖 TRAINING MODEL (NAIVE BAYES)...")
    print("=" * 80)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("✅ Model trained successfully!")
    print()
    
    # 6. Evaluate
    print("=" * 80)
    print("📊 EVALUATING MODEL...")
    print("=" * 80)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print()
    
    # 7. Visualize and save confusion matrix
    print("=" * 80)
    print("📊 CREATING CONFUSION MATRIX VISUALIZATION...")
    print("=" * 80)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['English', 'Indonesian'], 
                yticklabels=['English', 'Indonesian'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Language Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to static folder
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion Matrix saved as 'static/confusion_matrix.png'")
    plt.close()
    print()
    
    # 8. Save model and vectorizer
    print("=" * 80)
    print("💾 SAVING MODEL...")
    print("=" * 80)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Model saved as 'model.pkl'")
    print("✅ Vectorizer saved as 'vectorizer.pkl'")
    print()
    
    print("=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print("✅ Model saved as 'model.pkl'")
    print("✅ Vectorizer saved as 'vectorizer.pkl'")
    print("✅ Confusion Matrix saved as 'static/confusion_matrix.png'")
    print("\nYou can now run 'python app.py' to start the web server.")
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    train_and_save_model()
