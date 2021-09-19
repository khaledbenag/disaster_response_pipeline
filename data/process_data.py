
# import libraries
import sys
import pandas as pd
import numpy as np
import re
import sklearn as sk
import nltk
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    this function loads messages and categories data from a csv files.
    
    Parameters
    ----------
        messages_filespath : str
            the path to the messages csv file.
        categories_filepath : str
            the path to the categories csv file.
    
    Returns
    -------
        df : pandas dataframe
            the concatenation of messages and categories on "id".
    
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on = "id")
    
    return df


def clean_data(df):
    """
    this function cleans the df data to create 36 labels from categories, 
    convert columns values to numeric and drop nan and duplicates values.

    Parameters
    ----------
    df : pandas dataframe
        the messages and categories data in one file to be cleaned.

    Returns
    -------
    df : pandas dataframe
        a cleaned dataframe.

    """
    # split categories
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories.
    category_colnames = [x for x in row.str.split('-', expand = True)[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # get labels from string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop nan values 
    #TODO: impute missing values
    df.dropna(inplace = True)
    return df


def save_data(df, database_filename):
    """
    function to save data in sql database

    Parameters
    ----------
    df : pandas dataframe
        the dataset to be saved.
    database_filename : str
        the path where sql database is saved.

    Returns
    -------
    None.

    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterTable', engine, index=False)  


def main():
    """
    main function

    Returns
    -------
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()