#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:45:44 2018

@author: calvin
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
#from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


train = pd.read_csv('/Users/calvin/python/crime project/house_price_train.csv',encoding='gbk')
#初始設定----------------------------------------------------------------------
train_y=train["SalePrice"]
#all_data = pd.concat([train, test], ignore_index = True)

#train_x=list(train.columns)
#label="SalePrice"
train.drop(['SalePrice'],axis=1, inplace=True)

train=train.fillna(0)
train=pd.get_dummies(train)
#training set
train_x=train.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
#valid_x=valid_x.values
#valid_y=valid_y.values
#valid_y=valid_y.reshape(-1,1)



#標準化-------------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
#valid_x = scaler.transform(valid_x)
#test = scaler.transform(test)   

#PCA---------------------------------------------------------------------------

pca_num=0
delta=0.1

pca=PCA(n_components = 0.999999)
train_x=pca.fit_transform(train_x)
#valid_x=pca.transform(valid_x)
#test=pca.transform(test)
#dimention=train_x.shape[1]


#seperate the dataset----------------------------------------------------------
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
   

models = [Ridge()]

names = [ "Ridge"]

for name, model in zip(names, models):
    score = rmse_cv(model, train_x, train_y)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))     

grid(Ridge()).grid_get(train_x,train_y,{'alpha':[1,10,20,30,40,45,50,55,60,65,70,80,90]})



score = rmse_cv(Ridge(), train_x, train_y)
print(score)
    
    
    

#ridge = Ridge(alpha=60)
