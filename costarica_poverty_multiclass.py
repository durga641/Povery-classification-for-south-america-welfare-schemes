# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:40:30 2018

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



#x_train._get_numeric_data().to_csv("numeric_data.csv")
#x_train.info()
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
y_train=x_train['Target']
x_train=x_train.drop(columns=['Target'])

print(len(x_train),len(y_train))

x_train.to_csv("input_data.csv")

###########################################################

dt=tree.DecisionTreeClassifier()


kfold=model_selection.StratifiedKFold(n_splits=10,random_state=105648)

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

    
###################################################################
#############################################################
    
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
######################################################

##HYPER PARAMER TUNING FOR STACKING
#################################################################


##  RANDOM FOREST
rdt=RandomForestClassifier()
dt_grid = {'max_depth':list(range(9,15)), 'min_samples_split':list(range(11,20)), 'criterion':['gini','entropy']}
grid_tree_estimator = model_selection.GridSearchCV(rdt,dt_grid,cv=kfold,scoring='f1_macro')
grid_tree_estimator.fit(x_train, y_train)
print(grid_tree_estimator.grid_scores_)
print(grid_tree_estimator.best_score_)    ### so far best 64.59  ### next 82.2  ### 81.9 ## 82.4
print(grid_tree_estimator.best_params_)
print(grid_tree_estimator.)

rdf_best=grid_tree_estimator.best_estimator_

x_test['Target']=grid_tree_estimator.predict(x_test)





############################## ADABOOST

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[50,100,120,150],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,0.4,0.6]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="f1_macro", n_jobs= 4, verbose = 1)

gsadaDTC.fit(x_train,y_train)
print(gsadaDTC.best_score_) 
ada_best = gsadaDTC.best_estimator_

print(gsadaDTC.cv_results_)

##########  GRADIENT BOOSTING    ################

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="f1_macro", n_jobs= 4, verbose = 1)

gsGBC.fit(x_train,y_train)

GBC_best = gsGBC.best_estimator_
print(GBC_best)
print(gsGBC.best_score_)




################ SVM ########

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(x_train,y_train)

SVMC_best = gsSVMC.best_estimator_

print(SVMC_best)

# Best score
gsSVMC.best_score_

##################3 DRAWING LEARNING CURVES

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


###############################################

print(grid_tree_estimator.score(x_train, y_train))
grid_tree_estimator.cv_results_

test.to_csv("submission.csv", columns=['Id','Target'],index=False)

x_train=x_train.drop(index)
probable_outliers=x_train.select_dtypes(include=['float64']).columns.tolist()
sns.heatmap(x_train)
g = sns.heatmap(x_train[['v2a1', 'meaneduc', 'overcrowding', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned','Target']].corr(),annot=True, fmt = "0.1f", cmap = "coolwarm",linewidths=2)

print(probable_outliers)

g = sns.factorplot(x="refrig_computer",y="Target",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")

x_train['refrig_computer']=x_train['refrig'].map(str) + x_train['computer'].map(str)+x_train['v18q'].map(str)
train['refrig_computer'] =x_train['refrig_computer']

g = sns.factorplot(x="refrig_computer",y="Target",data=x_train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("poverty class")

sns.countplot(x="refrig",data=x_train)


    sns.countplot(x='v18q',hue='Target',data=x_train)

x_train['refrig'].describe()


sns.boxplot(y='v2a1',data=x_train)

sns.kdeplot(data=x_train['computer'])

pd.crosstab(index=x_train["Target"], columns=[x_train['computer']])

sns.factorplot(x="v2a1", hue='Target',data=x_train, kind="count", size=6) 
sns.countplot(x="computer", data=x_train) 
sns.jointplot(x="computer", y="Target", data=x_train)

image1=sns.FacetGrid(x_train, row="meaneduc", col='overcrowding',hue="Target").map(plt.scatter, "hacdor", "rooms").add_legend()
image1.savefig("plot1.png")






