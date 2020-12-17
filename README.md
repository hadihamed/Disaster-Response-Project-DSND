# Disaster Response Pipeline

## Project Overview

This project is to train a model for categorizing messages received by disaster response teams. The model is then deployed in a web-app, where we can also see visualizations of the training data the model was built on.

The model is trained on real-world data received from disaster response teams. 

## Details

### Instructions

To run the program:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
	*Press ctrl+c to quit and close app

3. Open a new terminal, and get path items for project web app:
	'env|grep WORK'

4. Replace https://SPACEID-3001.SPACEDOMAIN with the SPACEID and SPACEDOMAIN returned from step 3, navigate to website


### Files

process_data.py
- Loads training data from designated csv files
- Cleans data for ML processing
- Saves data to database

train_classifier.py
- Loads processed data from database
- Splits data and trains classification model
- Saves best classification model as Pickle file

run.py
- Initiates Flask Webapp for visualizing data in a dashboard
