"""
Created on Fri Jan 26 15:02:19 2018
@author: khalida Douibi
"""
import numpy as np
from Load_datasets import SimpleCross,Yeast
from LaCova_class import Lacova
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score
import time
import json

Bdd=Yeast()
Max_depth=4
minNoObj=5
Result_Exuc_BD={'LaCova_Yeast':[]}

data=Bdd[0]
labels=Bdd[1]
Nlabels=Bdd[2]
  
[X_train,Y_train,X_test,Y_test]=SimpleCross(data,labels,5)
    
Result_Exuc_dep_Lacova=[]   
dep=1   #to begin with depth=1
while dep <= Max_depth:
    max_depth=dep  
    #**************************************************************************
    Result_it=[]
    print('depth',dep)
    for it in range(len(X_train)):
        print('crosV',it)
    #Lacova *************************************************************
    #learning
        print('learning')
        start_lacovaL = time.time()
        lacova=Lacova(Nlabels, max_depth, minNoObj)
        lacova.fit(X_train[it], Y_train[it])      
        end_lacovaL = time.time()    
        time_lacovaL=end_lacovaL- start_lacovaL 
    #***testing**
        print('testing')
        start_lacovaT = time.time()
        Y_pred=lacova.predict(X_test[it])
        end_lacovaT = time.time()
        
        time_lacovaT=end_lacovaT-start_lacovaT
    
        Accuracy=accuracy_score(Y_test[it],np.array(Y_pred)) #Subset accuracy in multi-label
        Hamming_Loss=hamming_loss(Y_test[it],np.array(Y_pred))
        jaccard_similarity=jaccard_similarity_score(Y_test[it],np.array(Y_pred) )
    
        Result_it.append([Accuracy,Hamming_Loss,jaccard_similarity])
    Res_it=np.array(Result_it).mean(axis=0)
    Result_Exuc_dep_Lacova.append([Res_it[0],Res_it[1],Res_it[2],time_lacovaL,time_lacovaT])


#        del(lacova)#i instantiate new object for each depth
    dep=dep+1
Result_Exuc_BD['LaCova_Yeast'].append(Result_Exuc_dep_Lacova)
    
with open('LaCova_Yeast', 'w') as f:
    f.write(json.dumps(Result_Exuc_BD, indent=4))
    
#************************************************************************************************************************
#IDEA
#1
#To open Json dict to check results:
#with open('résultats_test_BR_CC_Lacova', 'r') as f:
#    sortie = json.load(f)    
#2
##To generate the decision tree directly in png/jpg from  dot without using cmd command:
#from subprocess import check_call
#check_call(['dot','-Tpng','graph.dot','-o','OutputFile_LaCovaDT.png'])    
    
    
    
    
