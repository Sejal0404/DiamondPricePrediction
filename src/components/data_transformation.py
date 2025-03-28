from sklearn.impute import SimpleImputer ## Handling missing values
from sklearn.preprocessing import StandardScaler ## Feature scaling
from sklearn.preprocessing import OrdinalEncoder #Ordinal Encoding
## pipelines - to automate the process of data preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

## Data Ingestion Config Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()  # Corrected to use DataTransformationconfig

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            logging.info('Starting to get the data transformation object')

            #Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z'] 

            #Define the custom ranking for each ordinal variable
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']  

            logging.info('Pipeline Initiated')
            logging.info('Defining numerical and categorical pipelines')

            ##Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            #Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent') ),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler()) ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info('Pipeline Completed')
            logging.info('Saving the preprocessor object to file')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Train DataFrame Columns: {train_df.columns.tolist()}')  # Log the columns of the DataFrame

            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')
            logging.info(f'Test DataFrame Columns: {test_df.columns.tolist()}')  # Log the columns of the DataFrame

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']  # Only keep the target column and 'id'

            ## features into dependent and independent features
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            target_feature_train_df = train_df[target_column_name]  # Added definition for target_feature_train_df

            input_feature_train_arr = preprocessing_obj.fit_transform(train_df.drop(columns=drop_columns, axis=1))  # Corrected variable name
            # Removed the restriction on features
            # input_feature_train_arr = input_feature_train_arr[:, [0, 1, 2, 3, 4, 5]]  # Ensure only expected columns are kept

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Added definition for input_feature_test_arr

            logging.info("Applying preprocessing object on training and testing datasets.") 

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
