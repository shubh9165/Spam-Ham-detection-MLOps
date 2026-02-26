import pandas as pd
import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineeringConfig:
    Feature_engineering_dir=os.path.join('artifacts','feature_engineering_data')
    train_feature_engineering_data_path=os.path.join(Feature_engineering_dir,'train_feature_engineering_data.csv')
    test_feature_engineering_data_path=os.path.join(Feature_engineering_dir,'test_feature_engineering_data.csv')

class FeatureEngineering:

    def __init__(self):
        self.feature_engineering_config=FeatureEngineeringConfig()

    def tfidf(self,train_df,test_df,max_features):
        logging.info("TFIDF process start")
        try:
            tfidf=TfidfVectorizer(max_features=max_features)

            X_train=train_df['text'].values
            y_train=train_df['target'].values
            X_test=test_df['text'].values
            y_test=test_df['target'].values

            X_train_tfidf=tfidf.fit_transform(X_train)
            X_test_tfidf=tfidf.transform(X_test)

            X_train_df=pd.DataFrame(X_train_tfidf.toarray())
            X_test_df=pd.DataFrame(X_test_tfidf.toarray())

            X_train_df['lable']=y_train
            X_test_df['lable']=y_test

            logging.info("TFIDF process end")

            return X_train_df,X_test_df
        

        except Exception as e:

            raise CustomException(e,sys)




    def initiate_feature_engineering(self):
        logging.info("Feature Engineering process start")
        try:

            train_processed_data=pd.read_csv('artifacts/data_preprocessing_data/train_processed_data.csv')
            test_processed_data=pd.read_csv('artifacts/data_preprocessing_data/test_processed_data.csv')
            
            train_processed_data['text'] = train_processed_data['text'].fillna('')
            test_processed_data['text'] = test_processed_data['text'].fillna('')
            
            logging.info("train and test processed data read sucessfully")

            train_df,test_df=self.tfidf(train_processed_data,test_processed_data,500)

            os.makedirs(self.feature_engineering_config.Feature_engineering_dir,exist_ok=True)

            train_df.to_csv(self.feature_engineering_config.train_feature_engineering_data_path,header=True,index=False)
            test_df.to_csv(self.feature_engineering_config.test_feature_engineering_data_path,header=True,index=False)

            logging.info("train and test feature engineering data saved")

            return (
                train_df,test_df
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj=FeatureEngineering()
    train_feature_engineering_data,test_feature_engineering_data=obj.initiate_feature_engineering()
