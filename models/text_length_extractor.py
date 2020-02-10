import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """ Class of the text length extractor used in the classifier model."""

    @staticmethod
    def get_length(text):
        """ Extract the length of a text.
        
            Args :
                text (str) : text
               
           Output : 
                (int) : number of words in the text
          """
        words = nltk.word_tokenize(text)
        words = [w for w in words if w not in stopwords.words("english")]
        return len(words)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_tagged = pd.Series(x).apply(self.get_length)
        return pd.DataFrame(x_tagged)
