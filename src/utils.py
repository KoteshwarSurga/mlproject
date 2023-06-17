import os
import sys

import numpy as np
import pandas as pd
import dill

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from exception import CustomException
from logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}
        
        for model_key,model_value in models.items():
            logging.info(f"The model name is '{model_key}'")
            para = param[model_key]
            gs = GridSearchCV(model_value,para,cv=3)
            gs.fit(x_train,y_train)

            model_value.set_params(**gs.best_params_)
            model_value.fit(x_train,y_train)
            #model_value.fit(x_train,y_train)
            y_train_pred = model_value.predict(x_train)
            y_test_pred = model_value.predict(x_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            print('the test_model_score-------->',test_model_score,model_key)
            report[model_key] = test_model_score            
        print('the report is---------->',report)   
        return report
            
    except Exception as e:
        raise CustomException(e,sys)
        
    
    
    
    
    
    