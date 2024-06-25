import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize text input.

    Args:
    text (str): Text to be tokenized.

    Returns:
    clean_tokens (list): List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data from SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_data', engine)

# Load the pre-trained model
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Render the main page with visualizations.

    Extract data for visuals and create JSON-encoded plotly graphs.

    Returns:
    Rendered HTML template for the main page with embedded graphs.
    """
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    related_counts = df.groupby('related').count()['message']
    related_names = ['related', 'not_related']
    
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Relation',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Is Related?"}
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """
    Handle user query and display model results.

    Use the model to predict classification for the user query.

    Returns:
    Rendered HTML template for the results page with classification results.
    """
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template with the classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Run the Flask app.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
