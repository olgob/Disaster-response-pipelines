"""
This file loads data from the SQLite database,
Splits the dataset into training and test sets,
Builds a text processing and machine learning pipeline,
Trains and tunes a model using GridSearchCV,
Outputs results on the test set,
Exports the final model as a pickle file.
"""

# import libraries
from sqlalchemy import create_engine
import pandas as pd
import pickle
import sys

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # download for lemmatization

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from .text_length_extractor import TextLengthExtractor


def load_data(database_filepath):
    """ Load the dataset.

        Args :
            database_filepath (str) : dataset filepath relative to this python file.

        Output :
            x (list) : list of messages used for modeling.
            y (list) : list of integer corresponding to the categories classification of the messages.
            labels (list) : labels of the different categories of the classifier.

    """
    engine = create_engine('sqlite:///' + database_filepath)  # DisasterResponse.db
    print(engine.table_names())
    table_name = engine.table_names()[0]
    df = pd.read_sql_table(table_name, con=engine)
    x = df['message'].values
    labels = list(df.columns[5:])
    y = df[labels].values
    return x, y, labels


def tokenize(text):
    """ Transform text into sequences tokenization.

        Args :
            text (str) : text to tokenize

        Output :
            clean_tokens (list) : tokens in lower case, without stopwords and lemmatized

    """
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


def build_model():
    """ Create a machine learning  pipeline that should take in the message column as input and output classification
    results on the other 36 categories in the dataset.
    MultiOutputClassifier is used for predicting multiple target variables.

        Args :
            None
        Output :
           None

    """
    # Create the machine learning pipeline
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_length', Pipeline([
                ('count', TextLengthExtractor())
            ]))

        ])),

        ('moclf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    # Optimize the hyperparameters using gridsearch
    parameters = {
        'moclf__estimator__n_estimators': [10, 100]
        # 'moclf__estimator__min_samples_leaf': [1, 2, 4],
        # 'moclf__estimator__min_samples_split': [2, 5, 10],
        # 'moclf__estimator__max_depth': [10, 100, None]
    }

    cv = GridSearchCV(model, param_grid=parameters, n_jobs=-1, cv=3, verbose=10)

    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """ Evaluate the model using test sets.

        Args :
            model : machine learning model
            X_test : Input of the test sets
            Y_test : Ouput of the test sets
            category_names : Names of the categories used for the classification

        Output :
            None

    """
    # Get predictions of the model on X_test
    y_pred = model.predict(x_test)
    # Calculate the accuracy of the model on test sets
    accuracy = (y_pred == y_test).mean()
    print("\n")
    print("Accuracy:", accuracy, "\n")
    print("\n")
    print("Precision, Recall, and F-1 score for each output category of the dataset:")
    print("\n")
    # Calculate precision and recall for each output category of the dataset
    for i, column_name in enumerate(category_names):
        print(column_name, classification_report(y_test[:, i], y_pred[:, i]))
    return


def save_model(model, model_filepath):
    """ Save the model.

            Args :
                model : machine learning model
                model_filepath : filepath of the saved model

            Output :
                None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(x_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
