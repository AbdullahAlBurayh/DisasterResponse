import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath (str): Filepath for the SQLite database.

    Returns:
    X (DataFrame): Feature variable (messages).
    Y (DataFrame): Target variables (categories).
    category_names (Index): Names of the target categories.
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('clean_data', con=engine.connect())
    X = df['message'].fillna('')  # Feature variable
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    """
    Tokenize and lemmatize text.

    Args:
    text (str): Text to be tokenized.

    Returns:
    clean_tokens (list): List of cleaned tokens.
    """
    if text is None:
        return []
    
    # Tokenize text
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, normalize case, and remove leading/trailing white space for each token
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and perform grid search.

    Returns:
    cv (GridSearchCV): Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to word count vectors
        ('tfidf', TfidfTransformer()),  # Transform word count vectors to TF-IDF features
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Classifier
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on test data and print classification report.

    Args:
    model (GridSearchCV): Trained model object.
    X_test (DataFrame): Test feature variables.
    Y_test (DataFrame): Test target variables.
    category_names (Index): Names of the target categories.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model (GridSearchCV): Trained model object.
    model_filepath (str): Filepath to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function that orchestrates the model training pipeline:
    - Load data
    - Build model
    - Train model
    - Evaluate model
    - Save model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
