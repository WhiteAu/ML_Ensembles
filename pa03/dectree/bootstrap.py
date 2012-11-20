"""
Bootstrapping utilities
"""
import numpy as np
import pandas as pd
import random as rd
import DTree

class BootStrap(object):

    def __init__(self, data, label, num=10):
        self.data = data
        self.label = label
        self.numbags = num
        
        self.bags = self.create_bags(self.data, self.numbags)
        self.trees = None
        #self.trees = create_trees(self.bags, self.data, self.label)

    def create_bags(self, data, num=20):
        """
        data is expected to be a pandas DataFrame
        
        returns a list of bootstrapped samples
        """
        N = len(data)
        
        bags = []
        for k in range(num):
            b = pd.DataFrame(columns=data.columns)
            print b.columns
            #b = []
            for n in range(N):
                rd.seed()
                idx = idx = rd.randint(0,N-1) #get random index
                #print data.ix[idx:idx, :]
                b = b.append(data.ix[idx:idx, :], ignore_index=True) 
            bags.append(b)
        return bags
    
    
        
def create_trees(bags, data, label):

    N = len(bags)
    trees = []
    for k in range(N):
        #b = pd.DataFrame(bags[k])
        #print b.columns
        t = DTree.get_tree(bags[k], label)
        trees.append(t)
        
    return trees
    
def f_score(truth, pred):
    tp = 0
    fp = 0
    fn = 0

    if len(truth) != len(pred):
        print 'mismatched truth/prediction labels in f-score: %d vs. %d'%(len(truth), len(pred))
        return None

    for i in xrange(len(truth)):
        if truth[i] == pred[i]:
            #print'truth[%d] was: %d, pred[%d] was: %d' %(i, truth[i], i, pred[i])
            tp += 1
        else: #check this, but I think this should hold...
            fp += 1
            fn += 1
    if tp+fn != 0:
        r = tp /float(tp+fn)
    if tp+fp != 0:
        p = tp / float(tp+fp)

    if (p + r) != 0:
        #print 'tp was: %d, fp was: %d fn was: %d'%(tp, fp, fn)
        return 2 * ((p * r)/float(p + r))
    else:
        print 'An error has occurred in f-score. Returning None'
        return None

            
    

def bootstrap_evaluate(truth_vals, pred_vals, num=20):
    """
    Meant for one-off evaluations on a dataset. No permutations are made

    Returns (Mean, StdDeviation) on success, and (None,None) on failure
    """        
    if (len(truth_vals) != len(pred_vals)):
        print 'Error: mismatched truth and prediction values: %d versus %d'%(len(truth_vals),len(pred_vals))
        return (None, None)
    
    N = len(truth_vals) #used for random prediction
    
    scores = []
    
    for k in range(num):
        truth = []#list of our true labels
        pred = []#list of our predicted labels
        
        for j in range(N): #N should be the length of truth/predicted vals
            rd.seed()#use system time as seed
            idx = rd.randint(0,N-1) #get random index
            #print 'idx was: %d'%idx
            truth.append(truth_vals[idx])
            pred.append(pred_vals[idx])
            #print'truth_vals[%d] was: %f, pred_vals[%d] was: %f' %(idx, float(truth_vals[idx]), idx, pred_vals[idx])
        scores.append(f_score(truth, pred))#append our f-score to the scores
    mean = np.mean(scores)
    std = np.std(scores)
    return (mean, std)