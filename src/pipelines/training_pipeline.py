import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer  # Import ModelTrainer

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    print(train_data_path, test_data_path)    
    # Instantiate ModelTrainer and initiate model training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)  # Train the model

    print(f"Preprocessor saved at: {preprocessor_path}")
