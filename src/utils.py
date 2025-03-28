import pickle
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """Saves an object to a file using pickle."""
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            #Train model
            model.fit(X_train,y_train)

            #Predict Testing Data
            y_test_pred = model.predict(X_test)

            #Get R2 Score for train and test data
            #train_model_score = r2_score(ytrain,y_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except  Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
