import os,sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from exception import CustomException
from components.data_transformation import DataTranformation,DataTransformationConfig
from components.model_trainer import ModelTrainerConfig,ModelTrainer
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass    ####### used to create class variables #######

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        print(self.ingestion_config)
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            ######## reading the dataset present reading from csv file, if you want to read from mysql,mongodb write a client code and call here
            df=pd.read_csv('notebook/data/stud.csv')
            
            logging.info("read the dataset as dataframe")
            print(os.path.dirname(self.ingestion_config.train_data_path))
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=True,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTranformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    #print(train_arr.shape,test_arr.shape)
    model_training = ModelTrainer()
    #print(model_training)
    print(model_training.initiate_model_trainer(train_arr,test_arr))

