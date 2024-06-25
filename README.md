# Disaster Response Pipeline Project

## Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)
6. [Repository](#repository)


## Installation <a name="installation"></a>
The code in this repository requires Python 3.x and the following Python libraries:
- pandas
- numpy
- sqlalchemy
- nltk
- scikit-learn
- pickle

You will also need to have NLTK data packages for tokenization and lemmatization:
```python
nltk.download('punkt')
nltk.download('wordnet')
```

Project Motivation <a name="project-motivation"></a>

The objective of this project is to build a machine learning pipeline to categorize disaster response messages. The project includes an ETL pipeline to process the data and a machine learning pipeline to train and evaluate a classifier.

File Descriptions <a name="file-descriptions"></a>
* data/
    * disaster_categories.csv: Categories dataset
    * disaster_messages.csv: Messages dataset
    * DisasterResponse.db: SQLite database with cleaned data
    * process_data.py: ETL pipeline script


* models/
    * train_classifier.py: Machine learning pipeline script


* app/
    * templates/
        * go.html: Placeholder for classification result
        * master.html: Main page of the web app
    * run.py: Flask web app script

Instructions <a name="instructions"></a>
1. Run ETL pipeline:
    * Process the data and store it in the SQLite database
        ``` python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```
2. Run ML pipeline
    * Train the classifier and save the model
        ``` python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl ```
3. Run the web app:
    * Launch the web app
        ``` python app/run.py ```
4. Open the web app:
    * Go to http://0.0.0.0:3001/ in your browser

Acknowledgements <a name="acknowledgements"></a>
* Thanks to Udacity for providing the project template and datasets.


Repository <a name="repository"></a>
* The code for this project can be found in the following GitHub repository:
https://github.com/AbdullahAlBurayh/DisasterResponse


Feel free to customize this template further to suit your project's specific needs. If you have any additional information or sections you'd like to include, please let me know!