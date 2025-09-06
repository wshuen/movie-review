import streamlit as st
import numpy as np
import joblib
import re, string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# -------------------------------
# Load model & vectorizer
# -------------------------------
model = joblib.load("svm_tfidf_80_20_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer_80_20.joblib")

# -------------------------------
# NLTK setup
# -------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- POS tag converter (map NLTK POS → WordNet POS) ---
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN   # Default to noun

# --- Replace rating patterns ---
def replace_ratings(text):
    # Handle n/n patterns (e.g., 7/8, 10/10, 2/5)
    def convert_fraction(match):
        num, denom = match.group().split("/")
        num, denom = int(num), int(denom)
        if denom > 0:
            percent = (num / denom) * 100
            if percent <= 25: return " terrible "
            elif percent <= 50: return " poor "
            elif percent <= 75: return " good "
            else: return " excellent "
        return ""
    
    text = re.sub(r"\b\d+/\d+\b", convert_fraction, text)

    # Handle star ratings (with/without space, assume max 5)
    def convert_stars(match):
        num = int(match.group(1))
        percent = (num / 5) * 100
        if percent <= 25: return " terrible "
        elif percent <= 50: return " poor "
        elif percent <= 75: return " good "
        else: return " excellent "
    
    text = re.sub(r"(\d+)\s*stars?", convert_stars, text, flags=re.IGNORECASE)                            

    # Handle percentages (e.g., 100%)
    def convert_percent(match):
        num = int(match.group(1))
        percent = num
        if percent <= 25: return " terrible "
        elif percent <= 50: return " poor "
        elif percent <= 75: return " good "
        else: return " excellent "
    
    text = re.sub(r"(\d+)%", convert_percent, text)
    return text

# --- Bigram extraction function ---
def extract_bigrams(tokens, top_n=20):
    """
    Extract top N bigrams from a list of tokens and return as compound tokens.
    """
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_measures = BigramAssocMeasures()
    top_bigrams = bigram_finder.nbest(bigram_measures.pmi, top_n)  # Use PMI to find collocations
    bigram_tokens = ['_'.join(bigram) for bigram in top_bigrams]
    # Combine unigrams and bigrams
    tokens_with_bigrams = tokens + bigram_tokens
    return tokens_with_bigrams

# --- Full Preprocessing Function ---
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Replace ratings
    text = replace_ratings(text)
    
    # 3. Remove HTML tags (replace with space)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # 4. Remove URLs (replace with space)
    text = re.sub(r"http\S+|www\S+", " ", text)                                                         
    
    # 5. Remove punctuation (replace with space instead of deleting)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # First cleanup: remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # 6. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 7. Tokenization
    tokens = word_tokenize(text)

    # 8. Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # 9. Lemmatization with POS tagging
    pos_tags = pos_tag(tokens)  # [('mentioned', 'VBD'), ('movie', 'NN'), ...]
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    # 10. Bigram extraction
    tokens = extract_bigrams(tokens, top_n=20)

    # 11. Final cleanup (remove extra spaces)
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()

    return text

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
        # Preprocess input (use the function you defined above)
        cleaned_input = preprocess_text(user_input)

        # Convert to TF-IDF using the loaded vectorizer
        vector_input = vectorizer.transform([cleaned_input])

        # Predict sentiment using the loaded model
        prediction = model.predict(vector_input)[0]

        # 🔹 Get decision score (distance from boundary)
        decision_score = model.decision_function(vector_input)[0]

        # 🔹 Convert to confidence percentage (sigmoid mapping)
        import numpy as np
        confidence = 1 / (1 + np.exp(-abs(decision_score))) * 100

        # Display result with confidence
        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color:#d4edda;padding:15px;border-radius:5px;">
                    <b>Sentiment:</b> Positive 🙂 <br>
                    <b>Confidence:</b> {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#f8d7da;padding:15px;border-radius:5px;">
                    <b>Sentiment:</b> Negative 🙁 <br>
                    <b>Confidence:</b> {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )




