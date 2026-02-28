import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score,accuracy_score
import json



class ModelEvaluationConfig:
    model_evaluation_dir=os.path.join(os.getcwd(),'reports')
    model_report_dir=os.path.join(model_evaluation_dir,'report.json')


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config=ModelEvaluationConfig()

    def evaluate_model(self,X_test,y_test,cls):
        logging.info('all score are calculating')
        try:
            y_pred=cls.predict(X_test)
            precision=precision_score(y_test,y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            auc=roc_auc_score(y_test,y_pred)

            metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
            }
            logging.info('all score are calculated')
            return metrics_dict
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_model_evaluation(self):
        logging.info("ModelEvaluation process are start")

        try:
            train_df=pd.read_csv('artifacts/feature_engineering_data/train_feature_engineering_data.csv')
            test_df=pd.read_csv('artifacts/feature_engineering_data/test_feature_engineering_data.csv')

            logging.info("train and test data are read sucessfully")

            file_path='Final_model/model.pkl'

            with open(file_path ,'rb') as file:
                model=pickle.load(file)
            
            
            logging.info("Model load sucessfully")

            X_test=test_df.iloc[:,:-1].values
            y_test=test_df.iloc[:,-1].values

            metrics=self.evaluate_model(X_test,y_test,model)

            os.makedirs(self.model_evaluation_config.model_evaluation_dir,exist_ok=True)

            with open(self.model_evaluation_config.model_report_dir, 'w') as file:
                json.dump(metrics, file, indent=4)

            logging.info("metics score are saved")

            return (
                self.model_evaluation_config.model_report_dir
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=ModelEvaluation()
    evaluate=obj.initiate_model_evaluation()