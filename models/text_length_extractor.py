import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextLengthExtractor(BaseEstimator, TransformerMixin):

    def get_length(self, text):
        words = nltk.word_tokenize(text)
        words = [w for w in words if w not in stopwords.words("english")]
        return len(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.get_length)
        return pd.DataFrame(X_tagged)
