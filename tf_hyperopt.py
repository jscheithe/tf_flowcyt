# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:12:40 2017

@author: jakob
"""

try:
    from tf_flowcyt import tf_flowcyt
except ImportError:
    import tf_flowcyt
    
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

def objective(args):
    args = {
        'TRAIN' : 'True',
        'nTestSet' : 1,
        'nAeHiddenLayers' : args['ae']['nAeHiddenLayers'],
        'nAeHiddenSizes' : args['ae']['nAeHiddenSizes'],
        'nMlpHiddenLayers' : args['mlp']['nMlpHiddenLayers'],
        'nMlpHiddenSizes' : args['mlp']['nMlpHiddenSizes']
#        'dAeLearningRate' : args['dAeLearningRate'],
#        'dMlpLearningRate' : args['dMlpLearningRate'],
#        'dMlpLearningRate' : args['dMlpLearningRate'],
#        'nBatchSize' : args['nBatchSize'],
#        'nMaxSteps' : args['nMaxSteps']
    }
    
    dicResults = tf_flowcyt.run(args)    
    
    return dicResults
    

def optimize():
    space = {
        'ae': hp.choice('nAeHiddenLayers', [
            {
            'nAeHiddenLayers' : 1,
            'nAeHiddenSizes' : [hp.randint('nAeHiddenSizes', 10) + 1],
            }
#            {
#            'nAeHiddenLayers' : 3,
#            'nAeHiddenSizes' : [hp.quniform('AeHiddenlayer1/3', 10, 30, 1), 
#                                hp.quniform('AeHiddenlayer2/3',  1, 10, 1),
#                                hp.quniform('AeHiddenlayer3/3', 10, 30, 1)]
#            }
            
#            },
#            {
#            'nAeHiddenLayers' : 5,
#            'nAeHiddenSizes' : [hp.quniform('AeHiddenlayer1/5', 200, 300, 1), 
#                                hp.quniform('AeHiddenlayer2/5',  10,  70, 1),
#                                hp.quniform('AeHiddenlayer3/5',   1,  10, 1),
#                                hp.quniform('AeHiddenlayer4/5',  10,  70, 1),
#                                hp.quniform('AeHiddenlayer5/5', 200, 300, 1)]
#            },
#            {
#            'nAeHiddenLayers' : 7,
#            'nAeHiddenSizes' : [hp.quniform('AeHiddenlayer1/7', 700, 1300, 1), 
#                                hp.quniform('AeHiddenlayer2/7', 300,  700, 1),
#                                hp.quniform('AeHiddenlayer3/7', 200,  300, 1),
#                                hp.quniform('AeHiddenlayer4/7',  10,   70, 1),
#                                hp.quniform('AeHiddenlayer5/7', 200,  300, 1),
#                                hp.quniform('AeHiddenlayer6/7', 300,  700, 1),
#                                hp.quniform('AeHiddenlayer7/7', 700, 1300, 1)]
#            }                   
            ]),
#        
        'mlp': hp.choice('nMlpHiddenLayers', [
            {
            'nMlpHiddenLayers' : 1,
            'nMlpHiddenSizes' : [hp.randint('nMlpHiddenSizes', 10) + 1],
            }
#            {
#            'nMlpHiddenLayers' : 3,
#            'nMlpHiddenSizes' : [hp.quniform('MlpHiddenlayer1/3', 10, 30, 1), 
#                                hp.quniform('MlpHiddenlayer2/3',  1, 10, 1),
#                                hp.quniform('MlpHiddenlayer3/3', 10, 30, 1)]
#            },
#            {
#            'nMlpHiddenLayers' : 5,
#            'nMlpHiddenSizes' : [hp.quniform('MlpHiddenlayer1/5', 200, 300, 1), 
#                                hp.quniform('MlpHiddenlayer2/5',  10,  70, 1),
#                                hp.quniform('MlpHiddenlayer3/5',   1,  10, 1),
#                                hp.quniform('MlpHiddenlayer4/5',  10,  70, 1),
#                                hp.quniform('MlpHiddenlayer5/5', 200, 300, 1)]
#            },
#            {
#            'nMlpHiddenLayers' : 7,
#            'nMlpHiddenSizes' : [hp.quniform('MlpHiddenlayer1/7', 700, 1300, 1), 
#                                hp.quniform('MlpHiddenlayer2/7', 300,  700, 1),
#                                hp.quniform('MlpHiddenlayer3/7', 200,  300, 1),
#                                hp.quniform('MlpHiddenlayer4/7',  10,   70, 1),
#                                hp.quniform('MlpHiddenlayer5/7', 200,  300, 1),
#                                hp.quniform('MlpHiddenlayer6/7', 300,  700, 1),
#                                hp.quniform('MlpHiddenlayer7/7', 700, 1300, 1)]
#            }                   
            ])
#
#        'dAeLearningRate': hp.choice('dAeLearningRate', [0.1, 0.01, 0.001]),
#        'dMlpLearningRate': hp.choice('dMlpLearningRate', [0.1, 0.01, 0.001]),
#        'nBatchSize': hp.choice('nBatchSize', [8, 16, 32, 64]),
#        'nMaxSteps': hp.choice('nMaxSteps', [3, 5, 10])
    }
    
    trials = Trials()
    
    best_model = fmin(objective,
                      space=space,
                      algo=tpe.suggest,
                      max_evals=100,
                      trials=trials)
    print(best_model)
    print(space_eval(space, best_model))

    
                  
if __name__ == '__main__':
    optimize()
    
