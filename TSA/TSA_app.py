import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib



# Streamlit app
st.title("Twitter Sentiment Analysis App")
st.write("This app predicts the sentiment of a given text.")

# Input text box
text_input = st.text_area("Enter a text:")

# Load the saved model and TF-IDF vectorizer
lr_classifier = joblib.load(r"logistic_classifier.joblib")
tfidf_vectorizer = joblib.load(r"tfidf_vectorizer.joblib")

def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)
    
    # Remove special characters, numbers, and punctuations, except for @ and #
    tweet = re.sub(r'[^a-zA-Z#@]', ' ', tweet)
    
    # Tokenize the tweet
    tokens = word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoin tokens into a single string
    processed_tweet = ' '.join(tokens)
    
    return processed_tweet

if st.button("Predict Sentiment (Logistic Regression)"):
    if text_input:
        # Preprocess the input text
        processed_text = preprocess_tweet(text_input)
        # Transform the input text using TF-IDF vectorizer
        input_vector = tfidf_vectorizer.transform([processed_text])
        # Predict sentiment using the loaded Logistic Regression classifier
        prediction = lr_classifier.predict(input_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Predicted Sentiment (Logistic Regression): {sentiment}")
