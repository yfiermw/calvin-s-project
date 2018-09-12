# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
import  tensorflow as tf
from pandas.core.frame import DataFrame
import JA_house as ja

train = pd.read_csv('house_price_train.csv')
test = pd.read_csv('house_price_test.csv')
all_data = pd.concat([train, test], ignore_index = True)

feature=list(train.columns)
label="SalePrice"
feature.remove("SalePrice")

#將label 換到第一行
SalePrice=all_data.pop("SalePrice")
all_data.insert(0,"SalePrice",SalePrice)

#完成初始設定-------------------------------------------------------------------
#data cleaning ----------------------------------------------------------

 
            
# print出label對於其他所有變數的相關係數
ja.Plot_Corr_Label(label, all_data)

#檢查缺值以及缺值的比例並刪除某些變數很少的丟失值
all_data=ja.Checkdelete_Na(feature,label, all_data)   

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    all_data[col].fillna("None", inplace=True)
    
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    all_data[col].fillna(0, inplace=True)
all_data['LotFrontage']=all_data.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

mean=np.mean(all_data["LotFrontage"])
all_data["LotFrontage"]=all_data["LotFrontage"].fillna(mean)

#檢查缺失值以及缺失值比例
ja.Checkde_Na(feature, all_data)

# feature engineering----------------------------------------------------------

all_data=pd.get_dummies(all_data)
#ja.Plot_Corr_01(all_data)
ja.Plot_Corr_Matrix(all_data)
#feature2=list(all_data.columns)
#for i in feature2:
#    ja.outlier(i,label,all_data)



# 建模start
#------------------------------------------------------------------------------
#A linear regression learning algorithm example using TensorFlow library.
#from __future__ import print_function

# 調整data，準備開始建模

# 調整data，準備開始建模


train=all_data[all_data[label].notnull()]
test=all_data[all_data[label].isnull()].drop(label,axis=1)
train_y=train[label]
train_x=train.drop([label],axis=1)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)

train_x=train_x.values
train_y=train_y.values
train_y=train_y.reshape(-1,1)
valid_x=valid_x.values
valid_y=valid_y.values
valid_y=valid_y.reshape(-1,1)


#建模code
rng = np.random

# Parameters
learning_rate = 0.0000001
training_epochs = 20
display_step = 2

n_samples = train_x.shape[0]

# tf Graph Input
X = tf.placeholder("float64",[None, 290],name="my_x")
Y = tf.placeholder("float64",name="my_y")

# Set model weights
W=tf.Variable(tf.random_normal([290, 1]),name="weight" )
W=tf.cast(W, tf.float64)
#W = tf.Variable(rng.randn(), name="weight")
#b = tf.Variable(tf.random_uniform([train_x.shape,1],dtype=tf.float64 ), name="bias")
#b = tf.Variable(tf.ones([1,1]))
#b=tf.Variable(np.random.normal(0,0.05,train_x.shape),name="bias")
#b=tf.cast(b, tf.float64)

# Construct a linear model
a=tf.matmul( X,W )
pred=tf.cast(a, tf.float64)
pred = tf.add(a, b)


# Mean squared error
#cost = tf.reduce_mean(tf.reduce_sum(tf.square(Y - pred), 
#                     reduction_indices=[1]))

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(n_samples)
#hypothesis = tf.sigmoid(tf.multiply(X, W) + b)
#cost=tf.reduce_mean(Y *( tf.log(cost + 1e-4)) + (1 - Y) * (tf.log(1 - cost+1e-4))) 
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            x = x.reshape(x.shape[0],1)
            x=np.transpose(x)
            y = y.reshape(y.shape[0],1)
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.pause(0.1)
    plt.show()