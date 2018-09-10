# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 12:11:20 2018

@author: Venkat Durga Rao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import os
from collections import Counter
import seaborn as sns  
from sklearn import tree
from sklearn import model_selection
from collections import Counter
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



os.chdir("E:\costarica_multiclass_poverty")

print  ('DURGA U ROCKS')

## READ THE TEST & TRAIN DATA

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

missing_cols=train.columns[train.isna().any()].tolist()


##v2a1 mean
#v18q1 mode
####rez_esc drop or new category 
##meaneduc mean
##SQBmeaned  mean
#############  IMPUTE MISSING VALUES  #########

mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(train[['v2a1','meaneduc','SQBmeaned']]) 
train[['v2a1','meaneduc','SQBmeaned']] = mean_imputer.transform(train[['v2a1','meaneduc','SQBmeaned']])

#### droping columns with object data type
x_train=train.drop(columns=['dependency','edjefa','edjefe','idhogar','Id','rez_esc','v18q1'])

x_train=x_train.drop_duplicates()

mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(test[['v2a1','meaneduc','SQBmeaned']]) 
test[['v2a1','meaneduc','SQBmeaned']] = mean_imputer.transform(test[['v2a1','meaneduc','SQBmeaned']])


x_test=test.drop(columns=['dependency','edjefa','edjefe','idhogar','Id','rez_esc','v18q1'])


x_train.select_dtypes(include=['float64']).columns.tolist()

############################# DETECT OUTLIERS ############################


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from all fields
    #x_train._get_numeric_data().to_csv("numeric_data.csv")
#x_train.info()
probable_outliers=x_train.select_dtypes(include=['float64']).columns.tolist()

Outliers_to_drop = detect_outliers(x_train,2,probable_outliers)


####################### DROP OUTLIERS#####################
x_train = x_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
x_train.to_csv("input_data.csv")

x_train=pd.read_csv("input_data.csv")
y_train=x_train['Target']
x_train=x_train.drop(columns=['Target'])


print(len(x_train),len(y_train))



x_train['age12above']=train['r4h2']+train['r4m2']-train['hogar_mayor']

x_train['bedrooms_to_rooms'] = x_train['bedrooms']/x_train['rooms']
    x_train['rent_to_rooms'] = x_train['v2a1']/x_train['rooms']
    x_train['tamhog_to_rooms'] = x_train['tamhog']/x_train['rooms']

############################################

dt=tree.DecisionTreeClassifier()


kfold=model_selection.StratifiedKFold(n_splits=10,random_state=105646)

random_state = 105678
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring ='f1_macro', cv = kfold, n_jobs=4))
     
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())   
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
    
    

print(cv_res)



#################################

train['age12above']=train['r4h2']+train['r4m2']-train['hogar_mayor']
train['age12above_percent']=((train['r4h2']+train['r4m2'])/(train['r4t3']))*100
train=train.drop(columns=['age12above_percent'])
train['age12above_percent']=((train['rooms'])/(train['hhsize']))*100

x_train['age12above']=-1*train['overcrowding']+train['meaneduc']+(-1*train['hogar_nin'])

train=train[train[['hogar_nin']] <=6 &  train[['hogar_adul']] <=7 ]

g = sns.heatmap(train[['overcrowding','meaneduc','age12above','v18q1','hogar_nin','hogar_adul','Target']].corr(),annot=True, fmt = "0.1f", cmap = "coolwarm",linewidths=2)

g = sns.factorplot(x="r4h2",y="Target",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")

g = sns.factorplot(x="r4m2",y="Target",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")

g = sns.factorplot(x="hogar_adul",y="Target",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")

sns.countplot(x='hogar_adul',data=train)


g = sns.factorplot(x="age12above_percent",y="Target",data=train,kind="bar", size = 20 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")




