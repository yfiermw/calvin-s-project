 #-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import mquantiles  #四分位數
import warnings
warnings.filterwarnings('ignore')



#0623  implement for Data_Processing Tool_01,  JohnnyHung
#   function( data)
#   1. Plot each features, for data
#   2. type quit, leave this interface
def Data_processing_01(data):
    #print(dir()),  this doesn't function well when run the code directly.
    #only functions ok, when using python -i ooo.py then , run > dir() ..

    #Start the dialog box
    while 1:
        function_str = "###############################################\n"
        function_str = function_str + "00_Enter your option:  (Q or q is quit!)\n"
        function_str = function_str + "02_Enter 'corr' to see the correlation!\n"
        function_str = function_str + "03_Enter 'see' to See Data structure!\n"
        function_str = function_str + "04_Enter 'check_na' to Check NA Value!\n"
        function_str = function_str + "05_Enter 'plot_relation'(my_feature,my_label)!\n"
        function_str = function_str + "06_Enter 'plot_corr_01' to see mix distribution!\n"
        function_str = function_str + "07_Enter 'plot_corr_matrix' to see matrix !\n"
        function_str = function_str + "###############################################\n"
        option = input(function_str)
        if option == 'q':
            print("Quit the JA_data_processing_01 function , Thanks!")
            break
        if option == 'Q':
            print("Quit the JA_data_processing_01 function , Thanks!")
            break
        if option == 'corr':
            print(data.corr())
        if option == 'see':
            print('The data structure is:  with option: ',option)
            print(data.head())
        if option == "check_na":
            print('Starting Checking NA.. with option:',option)
            my_list = list(data)
            Check_Na(my_list,data)
        if option == "plot_relation":
            print('Starting Plot Relation with option:',option)
            option_sub_01 = input("Input Features: ")
            option_sub_02 = input("Input Labels: ")
            Plot_Relation(option_sub_01,option_sub_02,data)
        if option == "plot_corr_01":
            Plot_Corr_01(data)
        if option == "plot_corr_matrix":
            Plot_Corr_Matrix(data)

        else:
            print('Your option is  ',option)

#畫圖且調整畫圖的幅度
def Plot(xs,ys,border=0.2):

    length_xs = 0.0
    length_ys = 0.0
    length_xs_max = 0.0
    length_xs_min = 0.0
    length_ys_max = 0.0
    length_ys_min = 0.0
    #Size
    length_xs  = max(xs) - min(xs)
    length_ys  = max(ys) - min(ys)
    
    length_xs_min = min(xs) - (length_xs * border)
    length_ys_min = min(ys) - (length_ys * border)
    length_xs_max = max(xs) + (length_xs * border)
    length_ys_max = max(ys) + (length_ys * border)

    print(length_xs_min,length_xs_max,length_ys_min,length_ys_max,'border(default = 0.2)',border)
    plt.axis([length_xs_min, length_xs_max,length_ys_min,length_ys_max])  #x 0-6,  y 0-8
    plt.plot(xs,ys,'ro',label='data') #red , point
    plt.legend(loc=4) #小圖示展示在右下側，loc=4為右下
    plt.show()
    
#------------------------------
# y 必須為array
# 測試learning rate、epoch關係圖程式
def Plot_loop(x, y, border=0.2):
    y=np.array(y)    
    length_xs  = max(x) - min(x)
    length_ys  = np.max(y) - np.min(y)
    color_list=['yellowgreen','gold','lightskyblue','lightcoral','blue','green','red']

    #print(min(y),max(y),'border(default = 0.2)',border)
    fig, ax = plt.subplots()
    for i in np.arange(y.shape[0]):
        for j in np.arange(y.shape[1]):
            
            if j+1<y.shape[1]:
                ax.scatter(x[j],y[i][j], color=color_list[i])      
            elif j+1==y.shape[1]:
                ax.scatter(x[j],y[i][j], color=color_list[i],label="row:%d" % i)
                
    plt.xlim((min(x)- (length_xs * border), max(x)+ (length_xs * border)))
    plt.ylim((np.min(y)- (length_ys * border), np.max(y)+ (length_ys * border)))    
    ax.set_xlabel('cost')
    plt.legend(loc=(1, 0))
    plt.show()
    plt.pause(0.1)







#檢查缺值以及缺值的比例並刪除某些變數很少的丟失值
#0.1可調整，表示丟失值比例佔0.1以下就刪除那些丟失值
def Checkdelete_Na(features_list,my_label, data):
    for each_col in features_list:
        if data[each_col].isnull().any().any():
            print(each_col,"has Na value: ")
            print (data[each_col].isnull().sum().sum())
            na_number = data[each_col].isnull().sum().sum()
            result = na_number /float(data[each_col].shape[0])
            print(result)
            if result <= 0.1:
               data=delete_NAN(each_col,data) 
               print("delete the feature:",each_col)
               print("-------------------")
            else: 
                print(" feature is ok:" , each_col)
                print("-------------------")
    return data

#檢查缺失值以及缺失值比例
def Check_Na(features_list, data):
    for each_col in features_list:
        if data[each_col].isnull().any().any():
            print("\n\n Na check start------------")
            print(each_col,"has Na value: ")
            print (data[each_col].isnull().sum().sum())
            na_number = data[each_col].isnull().sum().sum()
            result = na_number /float(data[each_col].shape[0])
            print(result)
        else: 
            print("there is no NA ")



#Note, you shall have three category
#1st:  Features with less uniques-elements ( <5) vs Label
#2nd:  Features is continues (like Age) vs Label
#3rd:  Features with much uniques-elements (Title,need to pre-processing , first) vs Label
#第一個變數: 想比較的feature，放自變數，他可能會有好幾種值，ex:1,2,3,4...
#第二個變數:因變數
#data之dataframe
            
#value count默认从最高到最低做降序排列
#dummy feature對因變數的關係
def Plot_Features_vs_Label(my_feature,my_label,my_data):
    plt.figure(1) #shall be increased to 2, 3, 4, ? we check later
    sns.barplot(x=my_feature, y=my_label, data=my_data, palette='Set3')
    my_list = my_data[my_feature].unique()
    for v in my_list:
        print("Percentage of "  +str(v) + ":")
        print("with  survived:%.2f \n" % (my_data[my_label][my_data[my_feature] == v].value_counts(normalize = True)[1]*100))
#注意你的data數據是否符合-> 上一行的[1]
#   print( "    百分比\n")
    print(my_data[my_label].value_counts(normalize = True)*100)
    plt.show()

#   my_feature:　自變數
#   my_label：　因變數
#   data：　ｒａｗ　ｄａｔａ
#畫圖-> 因變數對於自變數畫連續分布圖
def Plot_Relation(my_feature,my_label,data):
    facet1 = sns.FacetGrid(data, hue=my_label)
    facet1.map(sns.kdeplot,my_feature,shade= True)
    facet1.set(xlim=(0, data[my_feature].max()))
    facet1.add_legend()
    plt.show()

#多變數之間的散點圖 -> 輸出n*n矩陣圖
def Plot_Corr_01(data):
    pd.scatter_matrix(data,figsize=(6,6)) 
    plt.show()
    plt.pause(0.1)
    
#多變數之間相關係數顏色矩陣&print出來
def Plot_Corr_Matrix(data):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    aa=ax.matshow(data.corr())
    plt.xticks(range(len(data.columns)),data.columns)
    plt.yticks(range(len(data.columns)),data.columns)
    plt.colorbar(aa)
    print(data.corr())
    plt.show()
    plt.pause(0.1)
    
# print出label對於其他所有變數的相關係數
def Plot_Corr_Label(my_label,data):
    print("-------------------------")
    print("%s 對於其他變數的相關係數" % my_label )
    print(data.corr()[my_label]) 
    

#-------------------------------------------------以下功能未加入選單中
#刪除training set的NAN值
def delete_NAN(my_feature,data):
# 注意training set 和 testing set需要區分開
    data=data[data[my_feature].notnull()]
    return data

# 須設定回傳值，回傳值為新data
# 顯示離群值 & 刪除之
def outlier(my_feature,my_label,data):

    while 1:
   
        print("boxplot or normal distribution?\n")
        i=input("enter 'box'or 'normal' or 'end' to return data \n")     
        plt.ion()
        
        if i == "box":
            plt.figure(1)
            box=plt.boxplot(data[my_feature].values,showfliers=True)
            print(box)
            liers=box['fliers'][0].get_data()[1]
            print("-------------------")
            print(liers)
            plt.show()
            plt.pause(0.1)
            
##delete outliers
            print("delete outliers? Y/N?\n") 
            IQR = mquantiles(data[my_feature])[2] - mquantiles(data[my_feature])[0]
            maximun = mquantiles(data[my_feature])[2] + 1.5 * IQR
            print ('最大值',maximun)
            minimum = mquantiles(data[my_feature])[0] - 1.5 * IQR
            print ('最小值',minimum)
            data=data[(data[my_feature]<maximun) & (data[my_feature]>minimum)]
            break
            
            

        elif i == "normal":
            plt.figure(2)
            data[my_feature].plot(kind='kde',title='PDF')
            plt.show()
            plt.pause(0.1)
            
            print("-------------------")
            print ('超過最大值的異常值index', data[data[my_feature] > data[my_feature].mean() + 3 * data[my_feature].std()].index)
            print ('超過最大值的異常值', data[data[my_feature] > data[my_feature].mean() + 3 * data[my_feature].std()])
            print ('超過最小值的異常值index', data[data[my_feature] < data[my_feature].mean() - 3 * data[my_feature].std()].index)
            print ('超過最小值的異常值', data[data[my_feature] < data[my_feature].mean() - 3 * data[my_feature].std()])  
            
##delete outliers
            print("delete outliers? Y/N?\n") 
              
            data=data[(data[my_feature].mean() + 3 * data[my_feature].std() > data[my_feature]) & (data[my_feature] > data[my_feature].mean() - 3 * data[my_feature].std())]
            break
            
        elif i == "end":
            return data
        else:
            continue
        
#1. 刪除觀測樣本。
#2. 變換: 通過進行變量的變換也可以移除Outlier，例如對某個特徵全部取對數，一個值的自然對數可以減少極端值導致的大方差。
#3. 分檔: 例如，統計人的年收入和職業的關係時，像馬雲、貝克漢姆等人的收入就可以做一個分級，而不是直接使用數值，這樣就能避免數據太極端帶來的問題，相當於做了一個數據的離散化
#4. 填充值:使用平均值、中位數等值來替換Outlier的值，在填充之前我們應該分析該Outlier是自然存在的Outlier還是認為失誤導致的Outlier，如果是認為失誤導致的則可以進行這種處理，或者是使用其他模型預測的方式來填充。
#5. 把Outlier單獨處理，使用獨立的模型來應對這些Outlier。例如，我們分析身高和收入的關係時，可以考慮把從事籃球、排球、模特等職業的人員單獨拉個模型出來。


