import plotly.graph_objs as go

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.externals import joblib

from sqlalchemy import create_engine

import nltk

nltk.download('stopwords')
nltk.download('wordnet')  # download for lemmatization


def tokenize(text):
    """ Transform text into list of tokens with stopwords removed.
    
        Args :
            text (str) : Text
        
        Output :
            clean tokens (list) : Tokens with stopwords removed
            
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def get_data():
    """ Load data.
        
        Args :
            None
            
       Outputs :
            df : pandas dataframe containing the messages and their categories.
       
    """

    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse.db', engine)
    return df


def get_model():
    """ Load model.
        
        Args :
            None
            
       Outputs :
            model : model trained to classify messages previously trained.
       
    """
    model = joblib.load("../models/classifier.pkl")
    return model


def get_graphs():
    """ Creates two plotly visualizations.

        Args :
            None

        Output:
            graphs (list(dict)): list containing the four plotly visualizations

    """
    # Get the data
    df = get_data()

    # create visuals
    graphs = []

    # graph 1
    graph_one = []

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one.append(
        go.Bar(
            x=genre_names,
            y=genre_counts,
        )
    )

    layout_one = dict(title='Distribution of Message Genres',
                      xaxis=dict(title="Genre"),
                      yaxis=dict(title="Count"),
                      )

    # graph 2
    graph_two = []

    categories_names = df.columns[5:-1]
    categories_counts = df[categories_names].sum()

    graph_two.append(
        go.Bar(
            x=categories_names,
            y=categories_counts,
            marker=dict(color='red')
        )
    )

    layout_two = dict(title='Distribution of Message Categories',
                      xaxis=dict(tickangle=-45),
                      yaxis=dict(title="Count"),
                      )

    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))

    return graphs
