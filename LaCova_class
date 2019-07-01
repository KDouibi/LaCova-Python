Created on Tue Jan 16 15:30:07 2018
@author: Khalida Douibi
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import BinaryRelevance 
from skmultilearn.problem_transform import LabelPowerset
from collections import Counter
import math as m

class Lacova(object):

    def __init__(self, Nlabels, max_depth, minNoObj):
        self.Nlabels=Nlabels
        self.max_depth=max_depth
        self.minNoObj=minNoObj
    #******************************************************************************************************************   
    def fit(self, Train, Y_train):
        train=np.concatenate((Train, Y_train), axis=1)
        self.Attr=Train.shape[1] # features
        self.root = self.FindBestSplit_Opti(train)   
        self.split_f(self.root, 1) #1: initialisation depth 
        self.DOT_graph_tree(self.root)
        return self
    #******************************************************************************************************************           
    def FindBestSplit_Opti(self, D):
        
        minNoObj=self.minNoObj
        Attr=self.Attr  #features
        threshold=Threshold_lacova(D[:,Attr:])#self.threshold, the threshold should be computed based on labels
        Cut_Best=[]
        Di_best=[]
        labels_Leaf=[]
        
        #calculate the covariance matrix of D to decide if we make another split or apply single DT if the labels are indepandant or make the node as terminal(leaf)
        [SumOfCovariance,MatrixOfCovariance,SumOfVariance,VarianceElements]=self.Covariance(D[:,Attr:])
        if SumOfVariance==0 or len(D)<minNoObj:#labels are independant
                     #return leaf with relative frequencies of labels/stop growing the tree
                     #print('return leaf with relative labels /SumOfVariance==0 or len(D)<minNoObj or best_feat== -1')
                     #labels_Leaf=[]#self.Freq(D)#va etre faite dans split_f function                 
                    fbest=-2 #dans la fonction split_f quand il trouve f=-2 il applique freq pour avoir la feuille
                                                 
        else:
                    #labels are independant
                    if abs(SumOfCovariance)<= threshold: 
                        #print('Single DT')
                        #***************** BR DT *********************
                        try:
                            
                            classifier=self.BR_DT(D[:,0:Attr], np.array(D[:,Attr:]))
                            labels_Leaf=classifier                            
                            fbest=-1
                        except ValueError: 
                            return                                        
                    else: 
                        #labels are dependant                   
                        #print('find bestsplit(D)')
                        Qbest=380000 #initialization just for the first f
                        f=0
                        fbest=-99
                       
                        while f<Attr: #for each feature f
                            i=0
                            res=[]
                            
                            #print("Part 1: Find Best cut for the feature "+str(f))
                            while i<D.shape[0]: #if f is numerical attribute        
                                res.append(self.isanumber(D[i,f]))
                                i=i+1
                               
                            if np.any(res=='False'):
                                ###### Not functional, in progress #####
                                    #print("not numeric Split D into child nodes fDig according to values of f eg: yes/No")
                                    left,right= self.Split_categ(f,D)                
                            else:
                                    #print("FindBestCut for this attribute")
                                    outputsFindBestCut=self.FindBestCut_opti(D, f) ### on supprime Qbest  
                                    
                            Cut_Best_f= outputsFindBestCut['value']
                            #print("Cut_Best "+str(Cut_Best))
                            if Cut_Best_f== -99:
                                #Cut_Best_f== -99 means that no cut is good for the current attribute and we should consider another attribute for splitting, if all attribute were tested and no one is good for split, we consider the current node as a leaf
                                #f=-2 means that we should apply the function freq to annotate the leaf (see split_f ())
                               
                                
                                #print("attribute to be  ignored because all cuts are not interesting, so evaluation of another attribute") if all attributes are not interesting do no split so we should calculate the freq
                                if f== Attr and fbest==-99: #voir d.shape-1 !! garder que D.shape[1]-1
                                    fbest=-2
                                    labels_Leaf=self.Freq_LP(D)# je vais calculer freq dans split_f *******CHANGE
                                
                            else:
                            #split D according to Cut_best, calculates the Q and do that for each feature f   
                                #print("Part 2:  Evaluation of quality split if i use the feature "+str(f)+" with cut "+str(Cut_Best_f))
                                Q=outputsFindBestCut['qualite_split']
                                #print("quality"+ str(Q))
                                child_nodes=outputsFindBestCut['child_LR']
                                if Q <Qbest: 
                                    Qbest=Q  #Qbest: is the best quality of split reached at this level, Q: is the quality of split of the current feature f.
                                    fbest=outputsFindBestCut['index']
                                    Di_best=child_nodes
                                    Cut_Best=Cut_Best_f
                                    
                            f=f+1
        return {'best_feature':fbest,'cut_best':Cut_Best, 'childes_nodes':Di_best,'labels_Leaf':labels_Leaf} 
    #******************************************************************************************************************           
    def FindBestCut_opti(self, D, f):
        minNoObj=self.minNoObj
        Attr=self.Attr
        #print('sort D according to f in ascending manner')
        index_f=np.argsort(D[:,f]) #index of instances according to f in ascending manner
        f_sorted=D[index_f,f] #f sorted according to index
        #D_sorted=D[index_f,:] # D sorted according to index 
        #print('find all possible cut points Cut')
        [Cuts,indexes] = np.unique(f_sorted, return_index=True) #recuperate the possible unique cuts points without sort 
        cut=0     
        Qbest=38000
        Cut_best=-99
        left_best=[]
        right_best=[]
        while cut < Cuts.shape[0]:       
             #print("cut= %d " %cut)
             #print('split D into two child nodes D1 and D2 according to value of cut')        
             left, right,index_left,index_right=self.Split(f, Cuts[cut], D) 
             D1=left
             D2=right      
             #we should check that each child node has the number of instance > min (Change_here_idea!!), 
             #otherwise, the attribute f will be ignored (if all his cutpoints does'nt respect this condition we pass to another attribute)
             #if there is no attribute to consider, fbest= will be returned as -1 and so the function find best attribute will pass to another attribute and ignore the current one     
             if not D1 or not D2: # Check if list is empty
                 #print( "Empty" )
                 cut=cut+1 
             else:
                 #print( "Not empty" )
                 #calculate the quality of this split.
                 child_nodes=[D1,D2]                         
                 si1=np.array(D1).shape[0]
                 si2=np.array(D2).shape[0]                          
                 if si1 >= minNoObj and si2 >= minNoObj : 
                    
                     [SumOfCovariance1,MatrixOfCovariance1,SumOfVariance1,VarianceElements1]=self.Covariance(np.array(D1)[:,Attr:])
                     [SumOfCovariance2,MatrixOfCovariance2,SumOfVariance2,VarianceElements2]=self.Covariance(np.array(D2)[:,Attr:])
                     Q1=min([SumOfCovariance1,SumOfVariance1])
                     Q2=min([SumOfCovariance2,SumOfVariance2])
                     Qchild_nodes=[Q1,Q2]
                     #For each node calculate the quality value
                     Q=0
                     D_size = np.array(D).shape[0] #the original dataset
                     for i in range(len(child_nodes)):
                         D_i_size = np.array(child_nodes[i]).shape[0]
                         Q += (D_i_size/D_size)*Qchild_nodes[i]                     
                     #print("value of Q of child_nodes Cut %d"% Q) 
                     #print("Qbest/ Ancien %d"% Qbest)
                     if Q<Qbest: 
                         Qbest=Q
                         Cut_best=Cuts[cut] 
                         left_best=left
                         right_best=right
                     cut=cut+1
            
                 else:
                     #print("Cut point to ignored because the number of instances in child node < min")
                     cut=cut+1 
                     # to exit the loop and try another cut, since one of childes nodes does'nt have ennough instances                                  
        return {'index':f,'value':Cut_best,'child_LR':[left_best,right_best], 'qualite_split':Qbest}   
    #******************************************************************************************************************    
    def Split(self, f, Cut_best, D):
    #split function with indexes
       left, right = list(), list()
       #je dois récupérer les indexes of instances pour les utiliser dans matrix error et ne pas répéter l'application de BR sur les labels de gauche et à droite
       #mais directement récupérer les anciens 
       index_left,index_right=[],[]
       for inx,row in enumerate(D):
           if row[f] <= Cut_best:
               left.append(row)
               index_left.append(inx)
           else:
               right.append(row)
               index_right.append(inx)
                
       return left, right,index_left,index_right
     #******************************************************************************************************************   
      # Create child splits for a node or make terminal
      
    def split_f(self, node, depth):
        minNoObj=self.minNoObj
        max_depth=self.max_depth
        try:
            left, right = node['childes_nodes'] #recuperate the two childes nodes
            
        except ValueError: # error message because  can't make another split to reach the max depth value.
            #print(' can''t make another split to reach the max depth value. ')
            return
        best_feat=node['best_feature']
        del(node['childes_nodes']) #delete 'childes_nodes' from node in order to recuperate all children (left and right )  
        ##    	# check for a no split
        if not left or not right:
            sortie=self.Freq_LP(np.array(left + right))
            node['left'] = node['right']=sortie# important la classe la plus fréquente si je ne peux pas diviser encore ,
                                                                             #le noeud actuel sera une feuille et je fais la fréquence             
            return 
        
    	# check for max depth
        if depth >= max_depth:          
            node['left'], node['right']=self.Freq_LP(np.array(left)),self.Freq_LP(np.array(right))
            return
    	# process left child
        if len(left) <= minNoObj and best_feat != -1:
            node['left'] =self.Freq_LP(np.array(left))
        else:
            node['left'] = self.FindBestSplit_Opti(np.array(left))
            self.split_f(node['left'], depth+1)
    	# process right child
        if len(right) <= minNoObj and best_feat != -1:
            node['right'] =self.Freq_LP(np.array(right))
        else:
            node['right'] = self.FindBestSplit_Opti(np.array(right))
            self.split_f(node['right'], depth+1)
            return    
  #******************************************************************************************************************         
  #Freq: to create a terminal node values
    def Freq(self, D):
        a=self.Attr
        Label=[]
        while a< D.shape[1]:     
            val=Counter(D[:,a])
            most_com=val.most_common(1)[0][0] #1 is for the first tuple/ to recuperate the most common label
            Label.append(most_com) #Labels is the list of most frequent label in D
            a=a+1
        return Label      
  #******************************************************************************************************************            
  #*******************LP at leaves*****************  
    # freq with label powerset at leaves if labels are dependant:
    def Freq_LP(self, D):       
        a=self.Attr
        # initialize Label Powerset multi-label classifier
        # with a gaussian naive bayes base classifier
        classifier = LabelPowerset(DecisionTreeClassifier())#min_samples_split=self.minNoObj
        # train
        LP_classif=classifier.fit(D[:,0:a], D[:,a:])
        # predict
       # Label = classifier.predict(X_test)    
        return LP_classif      
  #******************************************************************************************************************    
  def predict(self, X_test):
        y_pred=[]
        root=self.root
        if root['best_feature'] == -1: #when we can't split at the begining
           #************ A fixer **********************************
            #When the root contains BR DT classifier, use the predict function 
            #works  just if the root is -1 but if i have mixture tree (Lacova +BR DT ), seecondition -1 in  recurse tree 
            #c'est le cas où dès le départ les labels sont indépendants car sumofCov< threshold donc directment j'applique BR sur D, mais dans
            #le cas où on dans les feuilles BR et d'autre LP/freq, celà va etre prédit dans la fonction recurse tree.
            #print('A')
            
            for t1 in (X_test):
                 y_pred_t=root['labels_Leaf'].predict(t1.reshape(1, -1)) ## error to fix when sumcov >thresholds
                 y_pred1=y_pred_t.toarray()#To pass from CSC matrix to array                 
                 pred=y_pred1[0,:].tolist()
                 y_pred.append(pred)                
        else:
            #pour parcourir l'arbre
            #print('B')
            
            for t in (X_test):
                pred=self.recurse_tree_LP(root, t)                
                #y_pred.append(pred[0])
                y_pred.append(pred) #best results with this
            return y_pred
            # Make a prediction with a decision tree 
  #******************************************************************************************************************            
  def recurse_tree(self, node, test):
        #rajouter la condition quand 
        if node['best_feature'] != -1: 
            if test[node['best_feature']] < node['cut_best']:
            
                if isinstance(node['left'], dict):  
                        return self.recurse_tree(node['left'], test)
                else:
                    return node['left']
            else:
                
                if isinstance(node['right'], dict):
                        return self.recurse_tree(node['right'], test)
                else:
                    return node['right']
        else:
            #print('Apply predict of BR DT')
            #on applique le classifieur BR qui a été stocké lors de l'apprentissage dans node['labels_leaf']
            pred_i=node['labels_Leaf'].predict(test.reshape(1, -1) )# i added [] to make 2D array for predict function of BR but i can use also .reshape(1, -1) 
            pred=pred_i.toarray() #### To pass from CSC matrix to dense
            pred=pred[0,:].tolist()          
            return pred 
  #******************************************************************************************************************   
  # LP on Leaves if labels are dependant
    def recurse_tree_LP(self, node, test):
        #rajouter la condition quand 
        if node['best_feature'] != -1: 
            if test[node['best_feature']] < node['cut_best']:          
                if isinstance(node['left'], dict):
                        return self.recurse_tree_LP(node['left'], test)
                else:#modifier pour s'adapter avec LP dans les feuilles
                    #on applique le classifieur LP qui a été stocké lors de l'apprentissage dans node['left']
                    pred_ii=node['left'].predict(test.reshape(1, -1) )# i added [] to make 2D array for predict function of BR but i can use also .reshape(1, -1) 
                    pred_l=pred_ii.toarray() #### To pass from CSC matrix to dense
                    pred_l=pred_l[0,:]# pour sortir de [[[]]] créer par reshape
                    return pred_l    
            else:
                #print('E')
                if isinstance(node['right'], dict):
                        return self.recurse_tree_LP(node['right'], test)
                else:#modifier pour s'adapter avec LP dans les feuilles 
                    #on applique le classifieur LP qui a été stocké lors de l'apprentissage dans node['right']
                    pred_i1=node['right'].predict(test.reshape(1, -1) )# i added [] to make 2D array for predict function of BR but i can use also .reshape(1, -1) 
                    pred_r=pred_i1.toarray() #### To pass from CSC matrix to dense
                    pred_r=pred_r[0,:]# pour sortir de [[[]]] créer par reshape
                    return pred_r
        else:
            #print('Apply predict of BR DT')
            #on applique le classifieur BR qui a été stocké lors de l'apprentissage dans node['labels_leaf']
            pred_i=node['labels_Leaf'].predict(test.reshape(1, -1) )# i added [] to make 2D array for predict function of BR but i can use also .reshape(1, -1) 
            pred=pred_i.toarray() #### To pass from CSC matrix to dense
            pred=pred[0,:]
                  
            return pred 
  #******************************************************************************************************************   
  # Print a decision tree
    def print_tree(self, node, depth=0):
        #reste a voir l'arbre de lacova comment ils le veulent ?et faire le print selon ca #problème quand les noeuds sont des classifieurs BR DT TO FIX      
            if isinstance(node, dict):
                if node['best_feature'] == -1:
                    print('[BR DT]')
                    return
                else:
                    print('%s[X%d < %.3f]' % ((depth*' ', (node['best_feature']), node['cut_best']))) #for each depth we print the child nodes (left & right)
                    self.print_tree(node['left'], depth+1)
                    self.print_tree(node['right'], depth+1)
            else:
                print('%s[%s]' % ((depth*' ', node)))#pour afficher la partie de l'arbre dessiné 
  #******************************************************************************************************************   
  def DOT_graph_node(self,f, node):
        
        if isinstance(node, dict) and 'best_feature' in node: #pour s'assurer que c'est un noeud pas une feuille car le dict dans la feuille ne coontient pas best_feature
            if node['best_feature'] == -1:
                        print('[BR DT]')
                        return
            else:
                
                        f.write(' %d [label="X%d < %.2f"];\n' % ((id(node), node['best_feature'], node['cut_best'])))
                        f.write(' %d -> %d;\n' % ((id(node), id(node['left']))))
                        f.write(' %d -> %d;\n' % ((id(node), id(node['right']))))
                        self.DOT_graph_node(f, node['left'])
                        self.DOT_graph_node(f, node['right'])
        else:
                
            khaloud = '[label ="'+ '\n'.join([str(key)+'='+str(value) for key, value in node.items()]) + '"] \n'            
            f.write(str(id(node)) + khaloud) 
  #******************************************************************************************************************   
  def DOT_graph_tree(self,node):
        fn = 'graph'+ '.dot'
        f = open(fn, 'w')
        f.write('digraph {\n')
        f.write('node[shape=rectangle];\n')#ellipse f.write('node[shape=rectangle];\n')
        self.DOT_graph_node(f, node)
        f.write('}\n') 
  #****************************************************************************************************************** 
  def Covariance(self,D):
        MatrixOfCovariance= np.cov(D);#Covariance matrix
        VarianceElements=np.diagonal(MatrixOfCovariance)
        SumOfVariance=np.nansum(abs(VarianceElements))# for ina in range(len(VarianceElements)) if np.isnan(VarianceElements[ina]) != True ) #calculates the sum of diagnal which are the variances
        SumOfCovariance= np.nansum(abs(np.triu(MatrixOfCovariance)))-SumOfVariance #si les valeurs sont différents de nan
        return SumOfCovariance, MatrixOfCovariance, SumOfVariance, VarianceElements 
  #****************************************************************************************************************** 
  def BR_DT(self,D,L):
        #DT usng BR of skmultilearn, to be compared with single_DT created manually in terme of predictions      
        classifier = BinaryRelevance(classifier=DecisionTreeClassifier())
        # train
        classifier.fit(D, L) 
        ### idea: predictions = classifier.predict_proba(X_test)
        return classifier 
  #****************************************************************************************************************** 
  def isanumber(self, a): 
        try:
            float(repr(a))
            bool_a = True
        except:
            bool_a = False   
        return bool_a      
  #******************************************************************************************************************      
  def Split_categ(self, f, D):
        #f is the categorical attribute 
        left, right = list(), list()
        unique_Elem=[]
        for wo in D[:,f]:
            if wo not in unique_Elem:
                unique_Elem.append(wo)
        print(unique_Elem)    
        for row in D:
               if row[f] == unique_Elem[0]: # only 2 values, for example: Yes or no
                                            # May be make another function for multi-class
                   left.append(row)
               else:
                   right.append(row)
               
        return left,right  
   #******************************************************************************************************************     
   def Threshold_lacova(D):
        #D are labels
        somme1=0
        somme2=0
        pi_=m.pi
        n=D.shape[0]
        L=D.shape[1]
        for j in range(L):
            pj=(Counter(D[:,j])[1])/n
            for k in range(j+1,L):
                #calculer la fréquence de label actifs pour j and k             
                pk=(Counter(D[:,k])[1])/n
                somme1+=m.sqrt(pj*pk*(1-pj)*(1-pk))
                somme2+=pj*pk*(1-pj)*(1-pk)
        Moy=m.sqrt(2/((n-1)*pi_))*somme1
        Ecart=m.sqrt(((1-(2/pi_))/(n-1))*somme2)
        threshold=Moy+(1.96*Ecart)      
        return threshold     
        
