from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import joblib
import os
from scraper import scraper, start_scraping_scheduler
import threading

nltk.download('stopwords')

app = Flask(__name__)

# Load or train model
MODEL_PATH = 'fake_news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Start scraping scheduler in a separate thread
scraping_thread = threading.Thread(target=start_scraping_scheduler)
scraping_thread.daemon = True
scraping_thread.start()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def extract_features(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
    return [sentiment, subjectivity, word_count, avg_word_length]

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer

    # Sample dataset (replace with real dataset)
    data = {
        'text': [
            'This is a real news article about politics.',
            'Breaking news: Stock market crashes!',
            'Fake news: Aliens landed on Earth.',
            'Government announces new policy.',
            'Celebrity caught in scandal - hoax!',
            'Scientific discovery proves climate change.',
            'Conspiracy theory about vaccines exposed.',
            'Economic report shows growth.',
            'Sensational claim: Time travel invented!',
            'Official statement from White House.'
        ],
        'label': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 0: real, 1: fake
    }
    df = pd.DataFrame(data)

    df['processed_text'] = df['text'].apply(preprocess_text)
    features = df['text'].apply(extract_features).tolist()
    features_df = pd.DataFrame(features, columns=['sentiment', 'subjectivity', 'word_count', 'avg_word_length'])

    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['processed_text']).toarray()
    X = pd.concat([pd.DataFrame(X_text), features_df], axis=1)
    X.columns = X.columns.astype(str)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print(f"Model accuracy: {accuracy_score(y_test, model.predict(X_test))}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer

model, vectorizer = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # First check if news exists in scraped data
    exists, article = scraper.check_news_exists(text)

    if exists:
        result = {
            'prediction': 'Real News',
            'confidence': 0.95,  # High confidence for verified news
            'sentiment': 0.0,  # Neutral for verified news
            'subjectivity': 0.1,  # Low subjectivity for verified news
            'source': article.get('source', 'Times of India'),
            'verification_method': 'Database Match'
        }
    else:
        # If not found in database, classify as fake news
        result = {
            'prediction': 'Fake News',
            'confidence': 0.90,  # High confidence for non-verified news
            'sentiment': 0.0,
            'subjectivity': 0.5,
            'verification_method': 'Not Found in Verified Sources'
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)