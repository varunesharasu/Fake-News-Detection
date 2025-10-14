# Fake News Detection Project

This is a complete fake news detection system built with Python, Flask, and machine learning.

## Features

- Real-time fake news detection
- Sentiment analysis
- Word construction analysis
- Web-based interface
- Machine learning model using Logistic Regression

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your browser and go to `http://127.0.0.1:5000/`

## Usage

1. Enter a news article in the text area
2. Click "Detect Fake News"
3. View the results including prediction, confidence, sentiment, and subjectivity

## How it Works

The system analyzes the input text using:
- TF-IDF vectorization for word features
- Sentiment analysis (polarity and subjectivity)
- Word count and average word length
- Logistic Regression classifier trained on a sample dataset

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Web interface
- `requirements.txt`: Python dependencies
- `fake_news_model.pkl`: Trained model (generated on first run)
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer (generated on first run)

## Note

This is a demonstration project using a small sample dataset. For production use, you should train the model on a larger, more diverse dataset.