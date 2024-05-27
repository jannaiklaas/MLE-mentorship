import re
import sys
import os
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from scipy.sparse import csr_matrix, save_npz, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'data_raw.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

def load_data(path):
    return pd.read_csv(path)

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    return train, test

def save_dataset(X, y, dir, filename) -> None:
    os.makedirs(dir, exist_ok=True)
    
    if isinstance(X, csr_matrix):
        # Convert y to a NumPy array and reshape to a 2D column vector
        y = y.to_numpy().reshape(-1, 1)
        
        # Combine the sparse matrix X and array y
        combined = hstack([X, csr_matrix(y)])
        
        # Save the combined sparse matrix
        path = os.path.join(dir, filename + ".npz")
        save_npz(path, combined)
    else:
        raise ValueError("Input X must be a csr_matrix")

class TextPreprocessor:
    def __init__(self, use_lemmatization=True, vectorization_type=None):
        self.use_lemmatization = use_lemmatization
        self.vectorization_type = vectorization_type
        self.stop_words = set(stopwords.words('english')) - {'not'}
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if not use_lemmatization else None
        self.vectorizer = None
        self.rare_words = None
        self.contraction_mapping = {
            r"\bdidn't\b": "did not", r"\bdon't\b": "do not",
            r"\bwasn't\b": "was not", r"\bisn't\b": "is not",
            r"\bweren't\b": "were not", r"\bare't\b": "are not",
            r"\bwouldn't\b": "would not", r"\bwon't\b": "will not",
            r"\bcouldn't\b": "could not", r"\bcan't\b": "can not",
            r"\bain't\b": "am not", r"\bdoesn't\b": "does not",
            r"\bshouldn't\b": "should not", r"\bhadn't\b": "had not",
            r"\bhaven't\b": "have not", r"\bhasn't\b": "has not",
            r"\bmustn't\b": "must not"
        }

    def preprocess(self, data, fit_vectorizer=False):
        if 'sentiment' in data.columns:
            y = data['sentiment'].map({'negative': 0, 'positive': 1})
        else:
            y = None
        
        if self.rare_words is None:
            self._calculate_rare_words(data['review'])

        X_cleaned = data['review'].apply(self._clean_text)

        if self.vectorization_type and fit_vectorizer:
            self.vectorizer = self._get_vectorizer()
            X_vectorized = self.vectorizer.fit_transform(X_cleaned)
        elif self.vectorization_type:
            X_vectorized = self.vectorizer.transform(X_cleaned)
        else:
            X_vectorized = X_cleaned

        return X_vectorized, y, self.vectorizer if fit_vectorizer else None

    def _get_vectorizer(self):
        if self.vectorization_type.lower() == 'ngrams':
            return CountVectorizer(ngram_range=(1, 3), stop_words=list(self.stop_words))
        elif self.vectorization_type.lower() == 'tf-idf':
            return TfidfVectorizer(stop_words=list(self.stop_words))
        else:
            raise ValueError("Invalid vectorization type specified.")

    def _initial_preprocess(self, text):
        for contraction, expanded in self.contraction_mapping.items():
            text = re.sub(contraction, expanded, text, flags=re.IGNORECASE)
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r"(n't|'d|'ll|'m|'re|'s|'ve|')", '', text, flags=re.IGNORECASE)
        tokens = word_tokenize(text)
        tokens = [word for word, pos in pos_tag(tokens) if pos not in ['NNP', 'NNPS']]
        tokens = [re.sub(r'\W+', ' ', word) for word in tokens if not word.isnumeric()]
        return [word.lower() for word in tokens]

    def _calculate_rare_words(self, reviews):
        all_words = [word for review in reviews for word in self._initial_preprocess(review)]
        word_counts = Counter(all_words)
        self.rare_words = {word for word, count in word_counts.items() if count == 1}

    def _clean_text(self, text):
        tokens = self._initial_preprocess(text)
        tokens = [word for word in tokens if word not in self.rare_words and word not in self.stop_words and len(word) > 2]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        else:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

def main():
    df = load_data(RAW_DATA_PATH)
    df = remove_duplicates(df)
    train, test = split_data(df)
    preprocessor = TextPreprocessor(use_lemmatization=True, vectorization_type="ngrams")
    X_train, y_train, _ = preprocessor.preprocess(train, fit_vectorizer=True)
    X_test, y_test, _ = preprocessor.preprocess(test, fit_vectorizer=False)
    save_dataset(X_train, y_train, PROCESSED_DATA_DIR, 'train_processed')
    save_dataset(X_test, y_test, PROCESSED_DATA_DIR, 'test_processed')

if __name__ == "__main__":
    main()