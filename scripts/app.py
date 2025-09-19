from flask import Flask, request, jsonify
import pickle

# --- Optional: NLTK nur, wenn du wirklich manuell tokenizest ---
# import nltk
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Modell laden (Pipeline mit Tfidf + LogisticRegression)
with open('models/logreg_spam_pipeline.pkl', 'rb') as f:
    logreg_pipeline = pickle.load(f)

BEST_THRESHOLD = 0.620

@app.route('/', methods=['GET'])
def home():
    return jsonify(status="ok", message="Spam API is running"), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Empfehlung: Rohtext direkt an die Pipeline geben,
    # damit Preprocessing identisch zu Training bleibt.
    prob = logreg_pipeline.predict_proba([text])[0][1]
    pred = int(prob >= BEST_THRESHOLD)
    label = 'spam' if pred == 1 else 'ham'
    return jsonify({'prediction': label, 'probability_spam': prob})

if __name__ == '__main__':
    # Debug ok, aber Reloader aus, damit nichts doppelt läuft
    app.run(debug=True, use_reloader=False)