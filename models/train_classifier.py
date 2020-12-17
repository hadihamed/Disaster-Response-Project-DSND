#All imports for script
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    
    """
    Description:
        Loads the data from specified databse
        
    Params:
        database_filepath (string): File path where database data is stored
        
    Returns:
        X (Dataframe): Dataframe of independent variables
        Y (Dataframe): Dataframe of output variables
        category_names (Array): Array of output variable names
    """
        
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table(database_filepath, engine)
    df.drop(columns = ['id', 'original','genre'], inplace = True)
    
    # Only keeping rows where related values are either 0 or 1
    df = df[df.related != 2]
    
    #Separating into X and Y dataframes
    X = df['message']
    Y = df.drop(columns = ['message'])
    category_names = Y.columns.values
    
    return X, Y, category_names


def tokenize(text):
    """
    Description:
        Tokenizes and Lemmatizes the text input, cleaning it of irregular characters and making entire string lowercase.
        
    Params:
        text (String): Text to be cleaned up and lemmatized
        
    Returns:
        tokens (Array): Array of lemmatized words in the sentence 
    """    
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    # Normalize case and remove puctuations
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    Description:
        Generates Pipeline for data transformation and modeling, including a cross-validation grid search to find the best model under the given parameter options.
        
    Params:
        None
      
    Returns:
        cv (Model): Multi-output model with the best parameters from cross-validation
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 20)))
    ])
    
    parameters = {'clf__estimator__max_features':['auto', None],
                  'vect__max_df': (0.5, 1.0)}

    cv = GridSearchCV(pipeline, parameters, scoring = 'f1_weighted', cv = 2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description:
        Evaluates model performance and prints summary statistics for each output metric.
    
    Params:
        model (Model): Multi-output model to be evaluated
        X_test (Numpy Array): Numpy array of Input feature(s)
        Y_test (Numpy Array): Numpy array of Output features
        category_names (Array): Array of output variable names
        
    Returns:
        None
    """    
    
    Y_pred = model.predict(X_test)
    scores = classification_report(Y_test, 
                                   Y_pred, 
                                   target_names=category_names)

    # print evaluation for each variable
    print("Testing scores for MultiOutputClassifier:")
    print(scores)


def save_model(model, model_filepath):
    """
    Description:
        Saves model to pickle file.
        
    Params:
        model (Model): Model to be saved
        model_filepath (String): filepath where the model will be stored
        
    Returns:
        None
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()