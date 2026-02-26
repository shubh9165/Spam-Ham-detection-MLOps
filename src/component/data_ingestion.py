import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import sys


@dataclass
class DataIngestionConfig:
    data_ingestion_dir=os.path.join('artifacts','data_ingestion_data')
    train_data_path=os.path.join(data_ingestion_dir,'train.csv')
    test_data_path=os.path.join(data_ingestion_dir,'test.csv')
    raw_data_path=os.path.join(data_ingestion_dir,'raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig();
        logging.info("Data Ingestion process start")

    def load_data(self)->pd.DataFrame:
        logging.info("spam.csv file reading process start")

        try:
            df=pd.read_csv(r'experiments/spam.csv')
            logging.info("spam.csv file read sucessfully")
            return df
        except Exception as e:
            raise CustomException(e,sys)


    def preprocessing(self,df:pd.DataFrame)->pd.DataFrame:
        logging.info('Data preprocessing start')

        try:
            df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
            df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
            logging.info('Data preprocessing completed')
            return df

        except Exception as e:
            raise CustomException(e,sys)

    def save_data(self,df:pd.DataFrame):
        
        logging.info("Process of saving train test model start")

        try:
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info("raw data saved sucessfully")

            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)
            logging.info("train data saved sucessfully")

            test_set.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)
            logging.info("test data saved sucessfully")


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            df=self.load_data()
            final_df=self.preprocessing(df)
            self.save_data(final_df)

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
