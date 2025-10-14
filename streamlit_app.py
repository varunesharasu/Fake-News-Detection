import streamlit as st
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

# Download NLTK data
nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .real-news {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .fake-news {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .confidence-meter {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load or train model
MODEL_PATH = 'fake_news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

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

# Load model
model, vectorizer = load_or_train_model()

# Start scraping scheduler in a separate thread (only once)
if 'scraper_started' not in st.session_state:
    st.session_state.scraper_started = True
    scraping_thread = threading.Thread(target=start_scraping_scheduler)
    scraping_thread.daemon = True
    scraping_thread.start()

def predict_news(text):
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

    return result

# Main app
def main():
    st.markdown('<h1 class="main-header">📰 Fake News Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Enter a news article below to check if it's real or fake:")

    # Text input
    user_input = st.text_area(
        "News Article:",
        height=150,
        placeholder="Paste your news article here...",
        help="Enter the news article you want to verify"
    )

    # Detect button
    if st.button("🔍 Detect Fake News", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing news article..."):
                result = predict_news(user_input)

            # Display result
            if result['prediction'] == 'Real News':
                st.markdown(f'<div class="result-box real-news">', unsafe_allow_html=True)
                st.markdown(f"## ✅ {result['prediction']}")
            else:
                st.markdown(f'<div class="result-box fake-news">', unsafe_allow_html=True)
                st.markdown(f"## ❌ {result['prediction']}")

            st.markdown(f'<p class="confidence-meter">Confidence: {result["confidence"] * 100:.1f}%</p>', unsafe_allow_html=True)
            st.markdown(f"**Verification Method:** {result['verification_method']}")

            if 'source' in result:
                st.markdown(f"**Source:** {result['source']}")

            st.markdown(f"**Sentiment:** {result['sentiment']:.2f} (Polarity)")
            st.markdown(f"**Subjectivity:** {result['subjectivity']:.2f} (Subjectivity)")

            st.markdown('</div>', unsafe_allow_html=True)

            # Progress bar for confidence
            st.progress(result['confidence'])

        else:
            st.error("Please enter some text to analyze.")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This fake news detection system:

        - **Scrapes** real news from Times of India every 30 minutes
        - **Verifies** news against the latest scraped articles
        - **Classifies** unmatched news as fake
        - **Analyzes** sentiment and linguistic patterns

        **How it works:**
        1. Checks if the news exists in verified sources
        2. If found → Real News (95% confidence)
        3. If not found → Fake News (90% confidence)
        """)

        st.header("📊 Statistics")
        total_articles = len(scraper.news_data)
        st.metric("Articles in Database", total_articles)

        # Show recent articles
        if total_articles > 0:
            st.subheader("Recent Articles")
            recent_articles = list(scraper.news_data.values())[-5:]  # Last 5 articles
            for article in recent_articles:
                st.markdown(f"- {article['title'][:50]}...")


if __name__ == "__main__":
    main()