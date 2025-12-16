"""
LANGUAGE DETECTION WEB APP - FLASK BACKEND
Kelompok 5 - Project Capstone Bu Erna

API Endpoints:
- POST /translate - Translate text and classify language
- GET / - Serve frontend
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
from deep_translator import GoogleTranslator

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global variables untuk model dan vectorizer
model = None
vectorizer = None

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model_and_vectorizer():
    """Load trained model and vectorizer from pickle files"""
    global model, vectorizer
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded successfully!")
        return True
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first using train_model.py")
        return False

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_language(text):
    """
    Predict language of input text
    
    Parameters:
        text (str): Input text
    
    Returns:
        dict: Prediction results with probabilities
    """
    if model is None or vectorizer is None:
        return None
    
    # Preprocess
    text_processed = text.lower()
    
    # Vectorize
    text_vec = vectorizer.transform([text_processed])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0]
    
    return {
        'language': prediction,
        'probabilities': {
            'English': float(proba[0]),
            'Indonesian': float(proba[1])
        },
        'confidence': float(max(proba)) * 100
    }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate text and classify language
    
    Request JSON:
        {
            "text": "input text",
            "source_lang": "id" or "en",
            "target_lang": "en" or "id"
        }
    
    Response JSON:
        {
            "success": true,
            "input_text": "...",
            "translated_text": "...",
            "input_classification": {...},
            "translated_classification": {...}
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        input_text = data['text'].strip()
        
        if not input_text:
            return jsonify({
                'success': False,
                'error': 'Text cannot be empty'
            }), 400
        
        source_lang = data.get('source_lang', 'id')
        target_lang = data.get('target_lang', 'en')
        
        # Translate text
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(input_text)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Translation error: {str(e)}'
            }), 500
        
        # Classify input text
        input_classification = predict_language(input_text)
        
        # Classify translated text
        translated_classification = predict_language(translated_text)
        
        return jsonify({
            'success': True,
            'input_text': input_text,
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'input_classification': input_classification,
            'translated_classification': translated_classification
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🌐 LANGUAGE DETECTION WEB APP")
    print("=" * 80)
    
    # Load model
    if load_model_and_vectorizer():
        print("\n🚀 Starting Flask server...")
        print("📍 Open http://localhost:5000 in your browser")
        print("=" * 80)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n⚠️  Please run 'python train_model.py' first to train and save the model.")
        print("=" * 80)
