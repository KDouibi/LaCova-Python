Created on Fri Jul 13 11:25:32 2018
@author: khalida Douibi


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold  
from operator import itemgetter

def Yeast():
    file_name="yeast_ss_attribut2.txt"
    Yeast_dataset=np.loadtxt(file_name)
    X=Yeast_dataset[:,0:103]# X predictors variables
    Y=Yeast_dataset[:,103:]#Y outcomes variables
    NLabels=Y.shape[1]
    #Threshold_lacova=0.26
    return X,Y,NLabels
    
    
def SimpleCross(X,Y,n_splits)
    kf= KFold(n_splits=n_splits, random_state=0)
    #Données
    X_train,XTr_ind=[],[]
    X_test,XTst_ind=[],[]
    #Labels
    Y_train=[]
    Y_test=[]
    for train_index, test_index in kf.split(X): 
        X_train.append(X[train_index,:])
        XTr_ind.append(train_index)
        Y_train.append(Y[train_index,:])
        
        X_test.append(X[test_index,:])
        XTst_ind.append(test_index)
        Y_test.append(Y[test_index,:])
    return  X_train, Y_train, X_test, Y_test
