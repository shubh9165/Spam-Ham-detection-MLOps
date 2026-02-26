import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.ensemble import RandomForestClassifier


class ModelBuildingConfig:
    model_dir = 'Final_model'
    model_path = os.path.join(model_dir, 'model.pkl')

class ModelBuilding:

    def __init__(self):
        self.model_building_config=ModelBuildingConfig()

    def model_training(self,X_train,y_train):

        try:
            logging.info('model training start')

            cls=RandomForestClassifier(n_estimators=22,random_state=2)

            cls.fit(X_train,y_train)
            logging.info('model training done')
            return cls


        except Exception as e:
            raise CustomException(e,sys)

    def save_model(self,file_path,model):
        try:
            with open(file_path,'wb') as file:
                pickle.dump(model,file)

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_building(self):
        logging.info('Model building process start')
        try:
            train_df=pd.read_csv('artifacts/feature_engineering_data/train_feature_engineering_data.csv')

            logging.info('sucesfully read feature engineering data')

            X_train=train_df.iloc[:,:-1].values
            y_train=train_df.iloc[:,-1].values

            cls=self.model_training(X_train,y_train)

            os.makedirs(self.model_building_config.model_dir,exist_ok=True);

            self.save_model(file_path=self.model_building_config.model_path,model=cls)

            logging.info("model sucessfully save")
            
            return self.model_building_config.model_path

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=ModelBuilding()
    model_fir=obj.initiate_model_building()
