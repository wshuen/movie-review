import streamlit as st
from joblib import load
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Preprocessing functions
# --------------------------
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords_and_lemmatize(text):
    final_text = []
    for word in text.split():
        word = word.strip().lower()
        if word not in stop and word.isalpha():
            lemma = lemmatizer.lemmatize(word)
            final_text.append(lemma)
    return " ".join(final_text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_square_brackets(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_stopwords_and_lemmatize(text)
    return text

# --------------------------
# Load model and TF-IDF
# --------------------------
svm_model = load('svm_model.joblib')
tfidf = load('tfidf_vectorizer.joblib')

# --------------------------
# Streamlit GUI
# --------------------------
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to see if it's Positive or Negative.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        # Preprocess input
        cleaned_input = denoise_text(user_input)

        # Convert to TF-IDF
        vector_input = tfidf.transform([cleaned_input])

        # Predict sentiment
        prediction = svm_model.predict(vector_input)[0]
        if prediction == 1:
            st.success("Sentiment: Positive :)")  
        else:
            st.error("Sentiment: Negative :(")   

