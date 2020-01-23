# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:21:23 2020

Pcut function computes the most frequent labels in a subset and assigns the final labels by considering the Label cardinality (LC) of the dataset.
For example, let consider a dataset with LC=4, first, Pcut computes the probability of an example to have each label in a subset of learning and sort them in a descending manner, and finally, it keeps four active labels with high probabilities (=LC).
This function has been used in many fields, you can find an example of use in a Personalized Decision Trees in our repository: LaCova-Python, where this function is used by the main authors to annotate the leaves. It has been used also for text categorization, as in [1]
[1] Yang, Y. (2001, September). A study of thresholding strategies for text categorization. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 137-145).

@author: khaloud
"""
import numpy as np
from collections import Counter

#D: dataset
#att= #input attributes
#LCard: Label Caardinality
#Nlabels: #labels

def Pcut(self,D):
        attr=self.Attr
        LCard=self.LC
        Nlabels=self.Nlabels
        nl=D.shape[0]
        index_L=[]
        confidence=[]
        a=attr
        while a< D.shape[1]:
            #For each label do:
            val=Counter(D[:,a])
            most_com=val.most_common(1)[0][0]               
            if most_com == 1:
                #computes probabilities of active labels (confidence)
                index_L.append(a-attr)#a-attr: to get the exact label index according to Nlabels of D.
                confidence.append(val.most_common(1)[0][1]/nl)
            a=a+1
      
        #initiate vector equal to Nlabels
        LFinal=np.zeros(Nlabels)
        if not index_L == False:
            #rank the active labels according to their probabilities  
            prob_labelsF=np.array(np.transpose([index_L,confidence]))
            sorteProbF = prob_labelsF[prob_labelsF[:,1].argsort()[::-1]] #[::-1]:for descending order
                
            indLFinal=[]
            lindfinal= len(index_L)
            LCard=int(LCard) 
            if LCard <= lindfinal:
                #if the golabl LCard is lower than the active labels predicted by the model.
            
                for l in range(LCard):
                    #recuperate the index positions of active labels.
                    indLFinal.append(sorteProbF[l][0] )
                #create the final label predictions.
                np.put(LFinal,indLFinal,1)
                
            else:
                for l in range(lindfinal):
                    indLFinal.append(sorteProbF[l][0] )
                        
                np.put(LFinal,indLFinal,1)                
        return LFinal
