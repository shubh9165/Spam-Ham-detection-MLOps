from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import yaml

class DataPreprocessingConfig:
    data_preprocessing_dir=os.path.join('artifacts','data_preprocessing_data')
    train_processed_data=os.path.join(data_preprocessing_dir,'train_processed_data.csv')
    test_processed_data=os.path.join(data_preprocessing_dir,'test_processed_data.csv')


class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config=DataPreprocessingConfig()
    @staticmethod
    def transform_text(text):
        """
        Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
        """
        ps = PorterStemmer()
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        text = nltk.word_tokenize(text)
        # Remove non-alphanumeric tokens
        text = [word for word in text if word.isalnum()]
        # Remove stopwords and punctuation
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        # Stem the words
        text = [ps.stem(word) for word in text]
        # Join the tokens back into a single string
        return " ".join(text)
    
    def load_params(self,params_path: str) -> dict:
        """Load parameters from a YAML file."""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.info('Parameters retrieved from %s', params_path)
            return params
        except Exception as e:
            raise CustomException(e,sys)
        
    def processed_df(self,df, text_column='text', target_column='target')->pd.DataFrame:

        try:
            logging.info('Starting preprocessing for DataFrame')
            # Encode the target column
            encoder = LabelEncoder()
            df[target_column] = encoder.fit_transform(df[target_column])
            logging.info('Target column encoded')

            # Remove duplicate rows
            df = df.drop_duplicates(keep='first')
            logging.info('Duplicates removed')
        
            # Apply text transformation to the specified text column
            df.loc[:, text_column] = df[text_column].apply(self.transform_text)
            logging.info('Text column transformed')
            return df
        


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_preprosessing(self):
        logging.info("Data Preprocessing process start")

        try:
            train_df=pd.read_csv('./artifacts/data_ingestion_data/train.csv')
            test_df=pd.read_csv('./artifacts/data_ingestion_data/test.csv')

            logging.info('Train and Test data are read sucessfully')

            train_processed_df=self.processed_df(train_df, "text", "target")
            test_processed_df=self.processed_df(test_df,"text","target")

            logging.info('Train and Test data Preprocessing done')

            os.makedirs(self.data_preprocessing_config.data_preprocessing_dir, exist_ok=True)

            train_processed_df.to_csv(self.data_preprocessing_config.train_processed_data,header=True,index=False)
            test_processed_df.to_csv(self.data_preprocessing_config.test_processed_data,header=True,index=False)

            logging.info('train and test processed data are saved')

            return (
                self.data_preprocessing_config.train_processed_data,
                self.data_preprocessing_config.test_processed_data
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj=DataPreprocessing()
    train_processd_data,test_processed_data=obj.initiate_data_preprosessing()