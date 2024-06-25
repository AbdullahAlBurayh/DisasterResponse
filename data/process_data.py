import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    df (DataFrame): Merged dataset containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataset.

    Args:
    df (DataFrame): Merged dataset containing messages and categories.

    Returns:
    df (DataFrame): Cleaned dataset with categories split into separate columns and duplicates removed.
    """
    # Split the categories
    categories = df.categories.str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[1]
    category_colnames = row
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        categories.rename(columns={column: column[:-2]}, inplace=True)
    
    # Correct the 'related' column name
    categories.rename(columns={'1': 'related'}, inplace=True)
    
    # Drop the original categories column from df and merge with new categories dataframe
    df = df.drop('categories', axis=1)
    df = pd.merge(df, categories, left_index=True, right_index=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an SQLite database.

    Args:
    df (DataFrame): Cleaned dataset.
    database_filename (str): Filepath for the SQLite database.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('clean_data', engine, index=False, if_exists='replace')
    print(df.shape[0])

def main():
    """
    Main function that orchestrates the ETL pipeline:
    - Load data
    - Clean data
    - Save data to database
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
