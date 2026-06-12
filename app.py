import streamlit as st
import joblib
import re
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Download required NLTK data for the web server silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="Spam Engine", page_icon="🛡️", layout="wide")

# =====================================================================
# 2. THE BLUEPRINTS: We must define the custom classes here 
# so joblib knows how to unpackage and run the model pipeline.
# =====================================================================
class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        features = []
        for text in X:
            text = str(text)
            cap_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
            url_count = len(re.findall(r'http[s]?://\S+', text))
            special_chars = len(re.findall(r'[!$#%]', text))
            features.append([cap_ratio, url_count, special_chars])
        return np.array(features)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    def fit(self, X, y=None): return self
    def transform(self, X):
        cleaned_text = []
        for text in X:
            text = re.sub(r'\W', ' ', str(text).lower())
            tokens = [self.stemmer.stem(w) for w in text.split() if w not in self.stop_words]
            cleaned_text.append(' '.join(tokens))
        return cleaned_text
# =====================================================================

MODEL_PATH = 'spam_filter_model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("The file does not exist at /content/")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model missing error: {e}")
    st.stop()

st.title("🛡️ NLP Spam Classification Engine")
st.markdown("Evaluate message integrity in real-time.")

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("Message Payload", height=200)
    analyze_btn = st.button("Execute Analysis", type="primary", use_container_width=True)

with col2:
    st.markdown("### Structural Telemetry")
    cap_metric, sym_metric, url_metric = st.empty(), st.empty(), st.empty()
    cap_metric.metric("Capitalization Ratio", "--")
    sym_metric.metric("Suspicious Symbols", "--")
    url_metric.metric("Hyperlink Count", "--")

if analyze_btn and user_input.strip():
    cap_ratio = sum(1 for c in user_input if c.isupper()) / (len(user_input) + 1)
    special_chars = len(re.findall(r'[!$#%]', user_input))
    url_count = len(re.findall(r'http[s]?://\S+', user_input))
    
    cap_metric.metric("Capitalization Ratio", f"{cap_ratio:.1%}")
    sym_metric.metric("Suspicious Symbols", special_chars)
    url_metric.metric("Hyperlink Count", url_count)
    
    with st.spinner("Analyzing high-dimensional vector space..."):
        # The pipeline automatically uses the custom classes defined above
        prediction = model.predict([user_input])[0]
        
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **THREAT DETECTED: SPAM**")
    else:
        st.success("✅ **CLEAN: HAM**")
