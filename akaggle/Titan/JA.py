from statistics import mean
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 





#xs = [1,2,3,4,5] #Feature
#ys = [5,4,6,5,6] #Label



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
    plt.legend(loc=4)
    plt.show()

def Plot_number(my_np,seperate_index):
    #my_image = my_np.reshape(28,28)
    my_image = my_np.reshape(seperate_index,seperate_index)
    im = plt.imshow(my_image, cmap="Greys")
    plt.show()



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
        option = raw_input(function_str)
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
            option_sub_01 = raw_input("Input Features: ")
            option_sub_02 = raw_input("Input Labels: ")
            Plot_Relation(option_sub_01,option_sub_02,data)
        if option == "plot_corr_01":
            Plot_Corr_01(data)
        if option == "plot_corr_matrix":
            Plot_Corr_Matrix(data)

        else:
            print('Your option is  ',option)



def Check_Na(features_list,data):
    for each_col in features_list:
        if data[each_col].isnull().any().any():
            print(each_col,"has Na value: ")
#            print data[each_col].isnull().sum().sum()
            na_number = data[each_col].isnull().sum().sum()
            result = na_number /float(data[each_col].shape[0])
            print (result)






#def Plot_Relation(my_feature,my_label,data):
#    facet1 = sns.FacetGrid(data, hue=my_label)
#    facet1.map(sns.kdeplot,my_feature,shade= True)
#    facet1.set(xlim=(0, train_data[my_feature].max()))
#    facet1.add_legend()
#    plt.show()
    

def Plot_Relation(my_feature,my_label,data):
    facet1 = sns.FacetGrid(data, hue=my_label)
    facet1.map(sns.kdeplot,my_feature,shade= True)
    facet1.set(xlim=(0, data[my_feature].max()))
    facet1.add_legend()
    plt.show()


def Plot_Corr_01(data):
    pd.scatter_matrix(data,figsize=(6,6))
    plt.show()
    

def Plot_Corr_Matrix(data):
    plt.matshow(data.corr())
    plt.xticks(range(len(data.columns)),data.columns)
    plt.yticks(range(len(data.columns)),data.columns)
    plt.colorbar()
    print(data.corr())
    plt.show()



#Note, you shall have three category
#1st:  Features with less uniques-elements ( <5) vs Label
#2nd:  Features is continues (like Age) vs Label
#3rd:  Features with much uniques-elements (Title,need to pre-processing , first) vs Label
def Plot_Features_vs_Label(my_feature,my_label,my_data):
    plt.figure(1) #shall be increased to 2, 3, 4, ? we check later
    sns.barplot(x=my_feature, y=my_label, data=my_data, palette='Set3')
    my_list = my_data[my_feature].unique()
    for v in my_list:
        my_str = "Percentage of " + str(v) + ":"
        print(my_str)
        print("with  Label:%.2f \n" % (data[my_label][my_data[my_feature] == str(v)].value_counts(normalize = True)[1]*100))
    plt.show()


    






#Reference coding for Plot_Relation
#    facet1 = sns.FacetGrid(train_data, hue="Survived")
#    facet1.map(sns.kdeplot,'Age',shade= True)
#    facet1.set(xlim=(0, train_data['Age'].max()))
#    facet1.add_legend()
#    plt.show()

