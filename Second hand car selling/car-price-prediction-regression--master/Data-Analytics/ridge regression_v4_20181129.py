
# coding: utf-8

# In[75]:


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



train = pd.read_csv('/Users/calvin/python/car project/testing_v1.csv')
test = pd.read_csv('/Users/calvin/python/car project/testing_v1.csv')

train=train[:800]
test=test.drop(test[:800].index,axis=0)

#def function(data, test):
    
train_y= train["Price"]
train_x= train.drop(["Price"],axis=1)
test_y= test["Price"]
test_x= test.drop(["Price"],axis=1)

train_x=pd.get_dummies(train_x)
test_x=pd.get_dummies(test_x)

#train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
test_x=test_x.values
test_y=test_y.values
test_y=test_y.reshape(-1,1)


test_y=test_y.astype(np.int)
train_y=train_y.astype(np.int)

# test_x, test_y -> testing set
# train_x -> training set & validation set feature
# train_y -> label of training and validation set




# In[84]:




# 標準化
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x=scaler.transform(test_x)


#PCA---------------------------------------------------------------------------
pca_num=0
delta=0.1
pca=PCA(n_components = 0.999999)
train_x=pca.fit_transform(train_x)
test_x=pca.transform(test_x)

dimension=train_x.shape[1]



# In[85]:


#train_y=train_y.values
#train_y=train_y.reshape(-1,1)  


# In[86]:


#    from sklearn import linear_model
#    K=3
#    kf = KFold(n_splits=K)
#    LR = linear_model.LinearRegression()
#    i=0
#    rmse=0
#    for train_index, test_index in kf.split(train_x):
#        LR.fit(train_x[train_index],train_y[train_index])
#        y_LR = LR.predict (train_x[test_index])
#    
#        plt.scatter(np.ones(y_LR.shape)*i,y_LR)
#        i=i+1
#        mse=np.mean((np.round(y_LR) != train_y[test_index])**2)
#        rmse+=np.sqrt(mse)/K
#    print(rmse)
#    print(LR.coef_)
#    print(LR.intercept_) 
#    
#    
#    
#    
#    y_LR_test = LR.predict (test_x)
#    print("y_LR_test =",y_LR_test)
#    # y_LR_test=np.mean(y_LR_test)
#    print("finish, plot the scatter figure")
#    plt.figure(3)
#    fig, ax = plt.subplots()
#    ax.scatter(y_LR_test, test_y)
#    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
#    ax.set_xlabel('Predicted')
#    ax.set_ylabel('label')
#    # plt.xlim((0,10))
#    # plt.ylim((0,10))
#    
#    plt.show()   
#    plt.pause(0.1)
#        
#    print(rmse)
#    print(LR.coef_)
#    print(LR.intercept_) 
#    

# In[87]:


#    #Linear Regression
#    from sklearn import linear_model
#    K=5
#    kf = KFold(n_splits=K)
#    LR = linear_model.LinearRegression()
#    i=0
#    rmse=0
#    
#    for train_index, test_index in kf.split(train_x):
#        LR.fit(train_x[train_index],train_y[train_index])
#        y_LR = LR.predict (train_x[test_index])
#    
#        plt.scatter(np.ones(y_LR.shape)*i,y_LR)
#        i=i+1
#        mse=np.mean((np.round(y_LR) != train_y[test_index])**2)
#        rmse+=np.sqrt(mse)/K
#    print(rmse)
#    print(LR.coef_)
#    print(LR.intercept_) 


# In[88]:


#kernel Ridge
degree=6
i=0
train_x_copy=train_x
for power in range(2,degree+1):
    i=i+1
    train_x=np.concatenate((train_x, np.power(train_x_copy,power)),axis=1) 

#kernelridge
from sklearn.linear_model import RidgeCV
K=5
kf = KFold(n_splits=K)
RR = RidgeCV(alphas=np.logspace(-3, 3, 100))
i=0
rmse=0
for train_index, test_index in kf.split(train_x):
    RR.fit(train_x[train_index],train_y[train_index])
    y_RR = RR.predict(train_x[test_index])

    plt.scatter(np.ones(y_RR.shape)*i,y_RR)
    i=i+1
    mse=np.mean((np.round(y_RR) != train_y[test_index])**2)
    rmse+=np.sqrt(mse)/K
    
y_RR = RR.predict (test_x)
#    print("y_LR_test =",y_RR)
#    # y_LR_test=np.mean(y_LR_test)
#    print("finish, plot the scatter figure")
#    plt.figure(3)
#    fig, ax = plt.subplots()
#    ax.scatter(y_RR, test_y)
#    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
#    ax.set_xlabel('Predicted')
#    ax.set_ylabel('label')
#    # plt.xlim((0,10))
#    # plt.ylim((0,10))
#    
#    plt.show()   
#    plt.pause(0.1)
#    print(RR.coef_)
#    print(RR.intercept_)
#    print(rmse)
#    print(RR.alpha_)
    
#return tesy_y  
  