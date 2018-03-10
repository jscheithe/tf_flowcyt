# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:12:40 2017

@author: jakob
"""

import math
import os
import time
import json
import matplotlib
matplotlib.use('agg') #required to run matplotlib on a machine without an X-server
import matplotlib.pyplot as plt

try:
    from tf_flowcyt import tf_flowcyt
except ImportError:
    import tf_flowcyt
    
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

def objective(args):
    args = {
        'TRAIN' : 'True',
#        'nTestSet' : 1,
        'nAutoencoders' : args['ae']['nAutoencoders'],
        'nAeSizes' : args['ae']['nAeSizes'],
        'nMlpHiddenLayers' : args['mlp']['nMlpHiddenLayers'],
        'nMlpHiddenSizes' : args['mlp']['nMlpHiddenSizes'],
#        'dRegBeta' : args['dRegBeta']
#        'dNoiseKeepProb' : args['dNoiseKeepProb']
#        'dAeLearningRate' : args['dAeLearningRate'],
#        'dMlpLearningRate' : args['dMlpLearningRate'],
#        'dMlpLearningRate' : args['dMlpLearningRate'],
#        'nBatchSize' : args['nBatchSize'],
#        'nMaxSteps' : args['nMaxSteps']
    }
    
    dicResults, _ = tf_flowcyt.run(args)    
    
    return dicResults
    

def optimize():
    space = {
#        'dRegBeta' : hp.loguniform('dRegBeta', math.log(0.0001), math.log(0.01))}
#        'nAutoencoders' : hp.choice('nAutoencoders', [0,1,2,3])}
#        'nAeSizes' : hp.choice('nAeSizes', [0, 0.000001, 0.00001, 0.0001, 0.001])}
#        'nMlpHiddenLayers' : hp.choice('nMlpHiddenLayers', [0, 0.000001, 0.00001, 0.0001, 0.001])}
#        'nMlpHiddenSizes' : hp.choice('nMlpHiddenSizes', [0, 0.000001, 0.00001, 0.0001, 0.001])}
#        'dNoiseKeepProb' : hp.loguniform('dNoiseKeepProb', math.log(0.0001), math.log(0.005))}
#        'dNoiseKeepProb' : hp.choice('dNoiseKeepProb', [0.0] + [2**(2*i) for i in range(-8, -2)])}
#        'nBatchSize' : hp.choice('nBatchSize', [16, 32, 64, 128, 265, 512])}
#        'dAeLearningRate' : hp.choice('dAeLearningRate', [0.001, 0.01, 0.1, 1])}
####################
        'ae': hp.choice('nAutoencoders', [
            {
            'nAutoencoders' : 0,
            'nAeSizes' : [],
            },            
            {
            'nAutoencoders' : 1,
            'nAeSizes' : [hp.randint('nAeHiddenSizes', 10) + 1],
            },
            {
            'nAutoencoders' : 3,
            'nAeSizes' : [hp.quniform('AeHiddenlayer1/3', 10, 30, 1), 
                                hp.quniform('AeHiddenlayer2/3',  1, 10, 1),
                                hp.quniform('AeHiddenlayer3/3', 10, 30, 1)]
            }
#################
            
#            {
#            'nAutoencoders' : 5,
#            'nAeSizes' : [hp.quniform('AeHiddenlayer1/5', 200, 300, 1), 
#                                hp.quniform('AeHiddenlayer2/5',  10,  70, 1),
#                                hp.quniform('AeHiddenlayer3/5',   1,  10, 1),
#                                hp.quniform('AeHiddenlayer4/5',  10,  70, 1),
#                                hp.quniform('AeHiddenlayer5/5', 200, 300, 1)]
#            }
#            {
#            'nAutoencoders' : 7,
#            'nAeSizes' : [hp.quniform('AeHiddenlayer1/7', 700, 1300, 1), 
#                                hp.quniform('AeHiddenlayer2/7', 300,  700, 1),
#                                hp.quniform('AeHiddenlayer3/7', 200,  300, 1),
#                                hp.quniform('AeHiddenlayer4/7',  10,   70, 1),
#                                hp.quniform('AeHiddenlayer5/7', 200,  300, 1),
#                                hp.quniform('AeHiddenlayer6/7', 300,  700, 1),
#                                hp.quniform('AeHiddenlayer7/7', 700, 1300, 1)]
#            }]                   
#############
            ]),
        
        'mlp': hp.choice('nMlpHiddenLayers', [
            {
            'nMlpHiddenLayers' : 1,
            'nMlpHiddenSizes' : [hp.randint('nMlpHiddenSizes', 15) + 1],
            },
            {
            'nMlpHiddenLayers' : 2,
            'nMlpHiddenSizes' : [hp.quniform('MlpHiddenlayer1/2', 1, 30, 1), 
                                hp.quniform('MlpHiddenlayer2/2',  1, 30, 1)]
            }                   
            ])
#####################
#
#        'dAeLearningRate': hp.choice('dAeLearningRate', [0.1, 0.01, 0.001]),
#        'dMlpLearningRate': hp.choice('dMlpLearningRate', [0.1, 0.01, 0.001]),
#        'nBatchSize': hp.choice('nBatchSize', [8, 16, 32, 64]),
#        'nMaxSteps': hp.choice('nMaxSteps', [3, 5, 10])
    }
    
    max_evals = 50
    
    print('hyperopt optimizing following parameters, max_evals=' + str(max_evals) + ':\n' + str(space) + '\n')
    
    trials = Trials()    
    best_model = fmin(objective,
                      space=space,
                      algo=tpe.suggest,
                      max_evals=max_evals,
                      trials=trials)
    print(best_model)
   
    s_identifier = 'tf_hyperopt.optimize() on ' + time.asctime()
    
    resultsdir = 'results/' + s_identifier + '/'
    os.mkdir(resultsdir)
    
    with open(resultsdir + 'trials.json', 'w') as f:
#        json.dump(trials.trials, f, indent=2)
        f.write(str(trials.trials))
       
    with open(resultsdir + 'best_model.json', 'w') as f:
        json.dump(best_model, f, indent=2)
        
#    fscore_overview = {}    
#    for para_name, para_vals in trials.vals.items():
#        fscores = [trials.results[i]['F1'] for i in range(len(trials.results))]
#        fscore_overview[para_name] = (para_vals, fscores)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        ax.scatter(para_vals, fscores) 
##        para_space = [0.01, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 1]
##        actual_para_vals = [para_space[idx] for idx in para_vals]
##        para_vals = actual_para_vals
#        
#        ax.scatter(para_vals, fscores)        
#        ax.set_ylabel('F1-Score')
#        fig.suptitle = para_name
#        fig.savefig(resultsdir + para_name + 'fscore_overview.png', dpi = 300)
#    
#    with open(resultsdir + 'fscores_overview.json', 'w') as f:
#        json.dump(fscore_overview, f, indent = 2)
        
    

    
    
if __name__ == '__main__':
    optimize()
    
