// ============================================================================
// DOM ELEMENTS
// ============================================================================

const inputText = document.getElementById('inputText');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const swapBtn = document.getElementById('swapBtn');
const translateBtn = document.getElementById('translateBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Translated text display
const translatedTextDisplay = document.getElementById('translatedTextDisplay');
const translatedLanguage = document.getElementById('translatedLanguage');
const translatedConfidence = document.getElementById('translatedConfidence');
const translatedProbId = document.getElementById('translatedProbId');
const translatedProbEn = document.getElementById('translatedProbEn');
const translatedProbIdValue = document.getElementById('translatedProbIdValue');
const translatedProbEnValue = document.getElementById('translatedProbEnValue');

// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Swap languages
swapBtn.addEventListener('click', () => {
    const temp = sourceLang.value;
    sourceLang.value = targetLang.value;
    targetLang.value = temp;
});

// Translate button
translateBtn.addEventListener('click', handleTranslate);

// Enter key in textarea (Ctrl+Enter to submit)
inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        handleTranslate();
    }
});

// ============================================================================
// MAIN FUNCTION
// ============================================================================

async function handleTranslate() {
    const text = inputText.value.trim();

    // Validation
    if (!text) {
        showError('Please enter some text to translate');
        return;
    }

    // Hide previous results and errors
    hideError();
    resultsSection.style.display = 'none';

    // Show loading state
    setLoading(true);

    try {
        // Call API
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                source_lang: sourceLang.value,
                target_lang: targetLang.value
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Translation failed');
        }

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// ============================================================================
// DISPLAY FUNCTIONS
// ============================================================================

function displayResults(data) {
    // Display translated text only
    translatedTextDisplay.textContent = data.translated_text;
    displayClassification(
        data.translated_classification,
        translatedLanguage,
        translatedConfidence,
        translatedProbId,
        translatedProbEn,
        translatedProbIdValue,
        translatedProbEnValue
    );

    // Show results section
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayClassification(
    classification,
    languageEl,
    confidenceEl,
    probIdEl,
    probEnEl,
    probIdValueEl,
    probEnValueEl
) {
    // Language badge
    languageEl.textContent = classification.language;

    // Confidence
    confidenceEl.textContent = `${classification.confidence.toFixed(2)}% confidence`;

    // Probabilities
    const idProb = classification.probabilities.Indonesian * 100;
    const enProb = classification.probabilities.English * 100;

    probIdEl.style.width = `${idProb}%`;
    probEnEl.style.width = `${enProb}%`;

    probIdValueEl.textContent = `${idProb.toFixed(2)}%`;
    probEnValueEl.textContent = `${enProb.toFixed(2)}%`;
}

// ============================================================================
// UI HELPER FUNCTIONS
// ============================================================================

function setLoading(isLoading) {
    if (isLoading) {
        translateBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
    } else {
        translateBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Check if backend is running
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (!data.model_loaded) {
            showError('Model not loaded. Please run train_model.py first.');
            translateBtn.disabled = true;
        }
    } catch (error) {
        showError('Cannot connect to backend. Please make sure the Flask server is running.');
        translateBtn.disabled = true;
    }
}

// Run health check on page load
checkHealth();
