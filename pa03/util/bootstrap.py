"""
Bootstrapping utilities
"""
import numpy as np
import pandas as pd
import random as rd

class BootStrap(object):

    def __init__(self, truth, pred, numfolds=10):
        self.truth_vals = truth
        self.prediction_vals = pred
        self.numfolds = numfolds

    def create_bags(self, data, num=self.numfolds):
        N = len(data)
        rd.seed()
        bags = []
        for k in range(num):
            b = []
            for n in range(N):
                idx = rd.random()%N #get random index

                b.append(data[idx])
            bags.append(b)
        return bags
        
    def evaluate(self, truth_vals, pred_vals, num=self.numfolds):
        
        if (len(truth_vals) != len(pred_vals)):
            print 'Error: mismatched truth and prediction values: %d versus %d'%len(truth_vals),len(pred_vals))
            return (None, None)
        
        N = len(truth_vals) #used for random prediction
        rd.seed()#use system time as seed
        scores = []
        
        for k in range(num):
            truth = []#list of our true labels
            pred = []#list of our predicted labels
            
            for j in range(N): #N should be the length of truth/predicted vals
                idx = rd.random()%N #get random index
                truth.append(truth_vals[idx])
                pred.append(pred_vals[idx])
            scores.append(f_score(truth, pred))#append our f-score to the scores

        mean = np.mean(scores)
        std = np.std(scores)
        return (mean, std)
                
def f_score(truth, pred):
    tp = 0
    fp = 0
    fn = 0

    if len(truth) != len(pred):
        print 'mismatched truth/prediction labels in f-score: %d vs. %d'%(len(truth), len(pred))
        return None

    for i in xrange(len(truth)):
        if truth[i] == pred[i]:
            tp += 1
        else: #check this, but I think this should hold...
            fp += 1
            fn += 1
    if tp+fn != 0:
        r = tp /float(tp+fn)
    if tp+fp != 0:
        p = tp / float(tp+fp)

    if (p + r) != 0:
        return 2 * ((p * r)/float(p + r))
    else:
        print 'An error has occurred in f-score. Returning None'
        return None

            
    

