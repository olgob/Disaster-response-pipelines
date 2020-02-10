import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load the data from 2 datasets.

        Args:
            messages_filepath (str) : .csv filepath corresponding to the messages dataset.
            categories_filepath (str) : .csv filepath corresponding to the categories dataset.

        Output:
            df : pandas dataframe corresponding to the merge datasets of categories and messages.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """ Clean the data realizing the following steps :
        - Split categories into separate category columns.
        - Use the first row of categories dataframe to create column names for the categories data.
        - Rename columns of categories with new column names.
        - Convert category values to just numbers 0 or 1.
        - Replace categories column in df with new category columns.
        - Remove duplicates.

        Args :
            df : pandas dataframe corresponding to the merge datasets of categories and messages.

        Output :
            df : cleaned pandas dataframe corresponding to the merge datasets of categories and messages.
        """
    # split the values in the categories column on the ; character so that each value becomes a separate column.
    regex = ";"
    categories = df['categories'].str.split(pat=regex, n=-1, expand=True)

    # use the first row of categories dataframe to create column names for the categories data.
    row = categories.transpose()[0]
    category_colnames = row.apply(lambda string: string[:-2]).values  # apply a lambda function that takes everything up
    # to the second to last character of each string
    # with slicing.

    # rename columns of categories with new column names.
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns.
    df = df.drop('categories', axis=1)  # drop the categories column from the df dataframe since it is no longer needed
    df = pd.concat([df, categories], axis=1)  # concatenate df and categories data frames.

    # remove duplicates.
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    return df


def save_data(df, database_filename):
    """ Save the clean dataset into an sqlite database.

        Args :
            df : cleaned pandas dataframe corresponding to the merge datasets of categories and messages.
            database_filename (str) : database filename.

        Output :
            None.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    """ Realize the ETL pipeline and create the database.

            Args :
                None.

            Output :
                None.
        """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
