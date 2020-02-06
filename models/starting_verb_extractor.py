import nltk
import pandas as pd
import re

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except:
                return False
        return False

    def tokenize(self, text):
        # convert to lowercase
        text = text.lower()
        # remove punctuation characters
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # words tokenization
        tokens = word_tokenize(text)
        # remove stopwords
        tokens_stopwords_removed = [w for w in tokens if w not in stopwords.words("english")]
        # reduce words to their root form performing Lemmatisation
        clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens_stopwords_removed]
        # reduce words to their word stem performing Stemming
        # clean_tokens = [PorterStemmer().stem(w) for w in tokens_stopwords_removed]
        return clean_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_transformed)
