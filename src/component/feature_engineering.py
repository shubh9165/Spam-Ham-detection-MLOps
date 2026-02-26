import pandas as pd
import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

class FeatureEngineeringConfig:
    Feature_engineering_dir=os.path.join('artifacts','feature_engineering_data')
    train_feature_engineering_data=os.path.join(Feature_engineering_dir,'train_feature_engineering_data.csv')
    test_feature_engineering_data=os.path.join(Feature_engineering_dir,'test_feature_engineering_data.csv')

class FeatureEngineering:

    def __init__(self):
        self.feature_engineering_config=FeatureEngineeringConfig()

    def initiate_feature_engineering(self,)