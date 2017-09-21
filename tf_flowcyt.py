# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:05:11 2016

@author: jakob
"""

import h5py
import numpy as np
import tensorflow as tf
import math
import time
import os
import argparse
#import matplotlib
#import matplotlib.pyplot as plt
import pandas
#from pandas.tools.plotting import scatter_matrix
import seaborn as sns
from hyperopt import STATUS_OK

try:
    from tf_flowcyt import helpers
except ImportError:
    import helpers
    


"""********DEFAULT PARAMETERS********"""
"""Train or load from disk?"""
TRAIN = True

"""Test Set Number"""
__n_testSet = 3

"""Hyperparameters"""
__n_ae_hidden_layers = 3
__l_ae_hidden_sizes = [75, 30, 75]
#__n_ae_hidden_layers = 5
#__l_ae_hidden_sizes = [150, 75, 30, 75, 150]

__n_mlp_hidden_layers = 2
__l_mlp_hidden_sizes = [30, 15]
#__n_mlp_hidden_layers = 4
#__l_mlp_hidden_sizes = [150, 75, 30, 15]

__activation_function = tf.nn.sigmoid
#__activation_function = tf.nn.relu

__n_classes = 2

__d_ae_learning_rate = 0.01
__d_mlp_learning_rate = 0.01

__n_batch_size = 64
__n_max_steps = 5


"""Paths"""
__s_datadir = 'data/'
__s_csvpath = 'exp_list_PR.csv'
__s_logdir = 'log/'
__s_model_save_path = 'data/models/'



def parse_args(args=None):
    """ args: dictionary of arguments (hyperparameters,...)"""
    
    global summary_args, TRAIN, __n_testSet, __n_ae_hidden_layers, __l_ae_hidden_sizes, __n_mlp_hidden_layers, __l_mlp_hidden_sizes, __d_ae_learning_rate, __d_mlp_learning_rate, __n_batch_size, __n_max_steps
    
 
    
    """If no parameters are passed as an argument, look for command line parameters."""
    if __name__ == "__main__" and args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('TRAIN', type=str, nargs='?', default=None,
                            help='train the model or load the last trained model from disk?')
        parser.add_argument('nTestSet', type=int, nargs='?', default=None,
                            help='number of the test set (for cross validation)')
        parser.add_argument('nAeHiddenLayers', type=int, nargs='?', default=None,
                            help='number of the hidden layers in the AE part of the model')
        parser.add_argument('nAeHiddenSizes', type=lambda s: [int(n) for n in s.split(',')], nargs='?', default=None,
                            help='number of nodes in the AE hidden layers, separated by commas: \',\'')
        parser.add_argument('nMlpHiddenLayers', type=int, nargs='?', default=None,
                            help='number of the hidden layers in the MLP part of the model')
        parser.add_argument('nMlpHiddenSizes', type=lambda s: [int(n) for n in s.split(',')], nargs='?', default=None,
                            help='number of nodes in the MLP hidden layers, separated by commas: \',\'')
        parser.add_argument('dAeLearningRate', type=float, nargs='?', default=None,
                            help='AE learning rate')
        parser.add_argument('dMlpLearningRate', type=float, nargs='?', default=None,
                            help='MLP learning rate')
        parser.add_argument('nBatchSize', type=int, nargs='?', default=None,
                            help='Batch Size for training')
        parser.add_argument('nMaxSteps', type=int, nargs='?', default=None,
                            help='maximum number of training iterations')
        
        argsNamespace = parser.parse_args()
        args = vars(argsNamespace)


    if args is not None:
            if args.get('TRAIN') is not None:    
                TRAIN = (args['TRAIN'] == 'True')
            if args.get('nTestSet') is not None:
                __n_testSet = args['nTestSet'] 
            if args.get('nAeHiddenLayers') is not None:
                __n_ae_hidden_layers = args['nAeHiddenLayers']
            if args.get('nAeHiddenSizes') is not None:
                __l_ae_hidden_sizes = args['nAeHiddenSizes']
            if args.get('nMlpHiddenLayers') is not None:
                __n_mlp_hidden_layers = args['nMlpHiddenLayers']
            if args.get('nMlpHiddenSizes') is not None:
                __l_mlp_hidden_sizes = args['nMlpHiddenSizes']
            if args.get('dAeLearningRate') is not None:
                __d_ae_learning_rate = args['dAeLearningRate']
            if args.get('dMlpLearningRate') is not None:
                __d_mlp_learning_rate = args['dMlpLearningRate']
            if args.get('nBatchSize') is not None:
                __n_batch_size = args['nBatchSize']
            if args.get('nMaxSteps') is not None:
                __n_max_steps = args['nMaxSteps']   
    
    #testing..
    #print(trainData[index_shuf[80]]-trainData_shuf[80])
    #print(trainLabels[index_shuf[80]]-trainLabels_shuf[80])
    #print(trainData[index_shuf[100]]-trainData_shuf[100])
    #print(trainLabels[index_shuf[100]]-trainLabels_shuf[100])
    #print(trainData[index_shuf[120]]-trainData_shuf[120])
    #print(trainLabels[index_shuf[120]]-trainLabels_shuf[120])
    
    
def loadData(files):
    data = np.empty((0,9))
    labels = np.empty((0,1))
    for filename in files:
        with h5py.File(filename, "r") as f:
            dsetX = f['data'][()]
            dsetY = f['label'][()]   
            data = np.concatenate((data, np.array(dsetX)))
            labels = np.concatenate((labels, np.array(dsetY)))
    return data, labels       
        

def load(nTestSetNumber):
    
    start_time = time.time()
    trainFiles, testFiles = helpers.createTrainTestH5(__s_csvpath, nTestSetNumber)
    duration = time.time() - start_time
    print("create/check h5 files: %.3f sec" % duration) 
    
    start_time = time.time()
    trainData, trainLabels  = loadData(trainFiles)
    duration = time.time() - start_time
    print("load train data: %.3f sec" % duration)   
    
    start_time = time.time()
    testData, testLabels = loadData(testFiles)
    duration = time.time() - start_time
    print("load test data: %.3f sec" % duration)    
    
    return trainData, trainLabels, testData, testLabels  

    
def variable_summaries(var, name):
  """Attaches a few summaries to a Tensor like e.g. the weights."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)    


def get_bias_variable(shape):
    return tf.get_variable('bias', shape, 
                           initializer=tf.constant_initializer(0.1))
    
    
def get_weights_variable(shape):
    return tf.get_variable('weights', shape, 
                           initializer=tf.random_normal_initializer(0, 1.0 / math.sqrt(float(n_features))))


def build_nn_layer(input_tensor, input_dim, output_dim, layer_name, act=__activation_function):
    """Note: for the addition, the bias value is broadcast to fit the batch size.
    """
    with tf.variable_scope('net'):
        bias = get_bias_variable([output_dim])
        weights = get_weights_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
        net = tf.matmul(input_tensor, weights) + bias
        
    if act is None:
        return net
    with tf.variable_scope('activation'):
        activation = act(net)
    return activation


def build_feedforward_network(input_tensor, hidden_layers, hidden_sizes, type='mlp'):
    """Builds an mlp network of arbitrary shape.
    input_tensor: here, the evaluated output of the ae network (tensor of shape [batch_size, n_features])
    The first layer recieves the ae network's output.    
    The hidden layers use sigmoid activation function, the last layer (for classification)
    gives unscaled output. Softmax activation is done within the loss function.
    todo: for summaries, tags are made unique by prepending the string type. This
          could be a problem if more than one network of the same type is used."""

    with tf.variable_scope('network'):
        activations = []
        for i in range(hidden_layers):  #we build the hidden layers
            with tf.variable_scope('layer' + str(i)):                
                full_layer_name = type + '_layer' + str(i)
                if i == 0:
                    new_layer = build_nn_layer(input_tensor, n_features, hidden_sizes[i], full_layer_name)  #better: replace n_features with tf.shape(input)[1], but tensor as argument for get_variable is not allowed
                    activations.append(new_layer)
                    continue                
                new_layer = build_nn_layer(activations[i-1], hidden_sizes[i-1], hidden_sizes[i], full_layer_name)                
                activations.append(new_layer)
        if type == 'ae':
            with tf.variable_scope('output'):
                output = build_nn_layer(activations[-1], hidden_sizes[-1], n_features, 'ae/output_layer')
        if type == 'mlp':             
            with tf.variable_scope('output'):
            #for an mlp, the output layer consists of __n_classes nodes and doesn't use an activation function (activation is calculated within the loss function)
                output = build_nn_layer(activations[-1], hidden_sizes[-1], __n_classes, 'mlp/output_layer', act=None)
        
            
                
#            with tf.variable_scope('layer' + str(i)):
#                if i == 0:
##                   #first layer
#                    bias = get_bias_variable([__l_mlp_hidden_sizes[i]])
#                    weights = get_weights_variable([n_features, __l_mlp_hidden_sizes[i]])
#                    variable_summaries(weights, 'mlp/layer' + str(i) + '/weights')                    
#                    input_values = input
#                    activations.append(tf.nn.sigmoid(tf.matmul(input_values, weights) + bias))
#                
#                elif i < __n_mlp_hidden_layers-1:
#                    bias = get_bias_variable([__l_mlp_hidden_sizes[i]])
#                    weights = get_weights_variable([__l_mlp_hidden_sizes[i-1], __l_mlp_hidden_sizes[i]])
#                    variable_summaries(weights, 'mlp/layer' + str(i) + '/weights')                    
#                    input_values = activations[i-1]    
#                    activations.append(tf.nn.sigmoid(tf.matmul(input_values, weights) + bias))
#                             
#                else:       
#                    #last layer
#                    bias = get_bias_variable([__n_classes])
#                    weights = get_weights_variable([ __l_mlp_hidden_sizes[-2], __n_classes])
#                    variable_summaries(weights, 'mlp/layer' + str(i) + '/weights')                           
#                    input_values = activations[i-1]
#                    
#                    #here, only linear activation of the output layer is computed,
#                    #softmax activation is computed within the loss function for
#                    #performance reasons
#                    y_ = tf.matmul(input_values, weights) + bias
                
                #for evaluation: compute the actual rate of correctly classified examples
    if type == 'mlp':
        with tf.name_scope('Evaluation'):
            with tf.name_scope('softmax_activation'):                
                softmax_output = tf.nn.softmax(output)
                for i in range(10):
                    tf.summary.scalar('softmax_output_class1' + str(i), softmax_output[i,1])
            
            with tf.name_scope('labels'):
                for i in range(10):
                    tf.summary.scalar('label' + str(i), y[i])
        
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(y, tf.cast(tf.argmax(softmax_output, 1), tf.float32))
                for i in range(10):
                    tf.summary.scalar('correct_prediction' + str(i), tf.cast(correct_prediction[i], tf.int64))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#                with tf.name_scope('accuracy'):
#                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
    return output

    
#build network
def build_network(x, y):
    start_time = time.time()
    with tf.variable_scope('ae'):
        ae_output = build_feedforward_network(x, __n_ae_hidden_layers, __l_ae_hidden_sizes, type = 'ae')
        ae_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae')    
        ae_loss = get_mse_loss(ae_output, x)
        with tf.variable_scope('AdamOptimizer'):
            ae_train_op = tf.train.AdamOptimizer(__d_ae_learning_rate).minimize(ae_loss, var_list = ae_var_list)
        
    with tf.variable_scope('mlp'):
        mlp_output = build_feedforward_network(ae_output, __n_mlp_hidden_layers, __l_mlp_hidden_sizes, type = 'mlp')
        mlp_loss = get_xentropy_loss(mlp_output, y)  
        mlp_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mlp')    
        with tf.variable_scope('AdamOptimizer'):
            mlp_train_op = tf.train.AdamOptimizer(__d_mlp_learning_rate).minimize(mlp_loss, var_list = mlp_var_list)
    
    output_tensor = mlp_output
    duration = time.time() - start_time
    print("build network: %.3f sec" % duration)
    return ae_loss, ae_train_op, mlp_loss, mlp_train_op, output_tensor


def get_mse_loss(input_tensor, output_tensor):
    with tf.name_scope('loss'):
        mse_loss = tf.reduce_mean(0.5*tf.square(tf.subtract(output_tensor, input_tensor)), name='mean_square_error')
        tf.summary.scalar('average_' + mse_loss.name, tf.reduce_mean(mse_loss))
    return mse_loss
        

def get_xentropy_loss(logits, labels):
    """Returns the loss tensor for the mlp network. 
    logits: batch output from the ae network
    labels: 1-D tensor containing the labels of the current batch 
            (length: __n_batch_size)"""
    with tf.name_scope('loss'):
        labels = tf.cast(labels, tf.int64)
        xentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
        tf.summary.scalar('average_' + xentropy_loss.name, tf.reduce_mean(xentropy_loss))
    return xentropy_loss


def get_ae_train_op(loss):
    return tf.train.AdamOptimizer(__d_ae_learning_rate).minimize(loss)


def get_mlp_train_op(loss):    
    return tf.train.AdamOptimizer(__d_mlp_learning_rate).minimize(loss)
    
    
def fill_feed_dict(n_batchnumber, data, labels):
    """Fills the feed_dict with a random batch of from trainData.
    Uses the shuffeled list of indices index_shuf to randomly draw training 
    data. Every value is only drawn once, as n_batchnumber runs from 0 to 
    n_events/__n_batch_size
    n_batchnumer < 0: fill feed dict with the whole set of training data (unshuffeled)."""
    
    if n_batchnumber < 0:
        labels_reshape = np.reshape(labels, np.shape(labels)[0])        
        feed_dict = {
            x: data,
            y: labels_reshape
        }
        return feed_dict
    
    n_start_index = n_batchnumber*__n_batch_size
    batch_data = np.zeros([__n_batch_size, n_features])
    batch_labels = np.zeros([__n_batch_size])
    for i in range(__n_batch_size):    
        batch_data[i,:] = data[index_shuf[n_start_index + i], :]
        batch_labels[i] = labels[index_shuf[n_start_index + i]]
        
    feed_dict = {
        x: batch_data,
        y: batch_labels
    }    
    return feed_dict    


def train(sess, train_op, loss):
    global global_step
    batches = math.floor(n_events/__n_batch_size)
    for step in range(__n_max_steps):
        start_time = time.time()
        loss_value = 0.0
        for batch_number in range(batches):            
            #get the feed_dict
            feed_dict = fill_feed_dict(batch_number, trainData, trainLabels)
            
            #perform training step
            _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            loss_value += np.mean(batch_loss)
            
        loss_value /= batches
        duration = time.time() - start_time            
        
        # Write the summaries and print an overview fairly often.
        if step % 1 == 0:
            # Print status to stdout.
            print('%s: Step %d: average: %.7f (%.3f sec)' % (loss.name, step, loss_value, duration))
            # Update the events file with results from the whole data set.
            full_feed_dict = fill_feed_dict(-1, trainData, trainLabels)
#            with tf.variable_scope('mlp', reuse=True):
#                with tf.variable_scope('Evaluation', reuse=True):
#                    with tf.variable_scope('accuracy', reuse=True):
#                        accuracy = tf.get_variable('accuracy')
#                        tf.scalar_summary('mlp/Evaluation/accuracy/accuracy', accuracy)
            summary_str = sess.run(summary_op, feed_dict=full_feed_dict)
            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()
            global_step += 1    
    return loss_value
            
def test(sess, logits):
    predictions_tensor = tf.nn.softmax(logits)
    
    feed_dict = fill_feed_dict(-1, testData, testLabels)
    
    predictions = sess.run(predictions_tensor, feed_dict = feed_dict)
    predicted_labels = np.reshape(np.argmax(predictions,1), (-1, 1))
    
#    correct_prediction = np.equal(testLabels, predicted_labels)
#    accuracy = sum(correct_prediction)/len(correct_prediction)
#    
#    print('test accuracy: %.3f' % (accuracy))
    
    return predicted_labels
    
    
def train_and_test():
    """
    globals:
        summary_writer
        ae_train_op
        mlp_train_op
        ae_loss
        mlp_loss
        trainData
        trainLabels
        global_Step
    """
        
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        summary_writer.add_graph(sess.graph, global_step=global_step)
        
        #ae_saver = tf.train.Saver(var_list = ae_var_list)
        #mlp_saver = tf.train.Saver(var_list = mlp_var_list)
        saver = tf.train.Saver()        
        if TRAIN == True:
            start_time = start_time_train = time.time()
            train(sess, ae_train_op, ae_loss)
            duration = time.time() - start_time
            print("AE training: %.3f sec" % duration)
            #ae_saver.save(sess, __s_model_save_path + 'ae_' + time.strftime('%Y%m%d-%H%M%S'))    
            
            start_time = time.time()
            train(sess, mlp_train_op, mlp_loss)
            duration = time.time() - start_time
            print("MLP training: %.3f sec" % duration)
            
            duration_train = time.time() - start_time_train
            print('train (overall)): %.3f' % duration_train)
            
            
            #mlp_saver.save(sess, __s_model_save_path + 'mlp_' + time.strftime('%Y%m%d-%H%M%S'))    
            
            saver.save(sess, __s_model_save_path + time.strftime('%Y%m%d-%H%M%S'))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(__s_model_save_path))
        
        start_time = time.time()
        test_predictions = test(sess, output_tensor)
        duration = time.time() - start_time
        print("test: %.3f sec" % duration)
        return test_predictions
    
    
def plot_predictions(data, predictions, plotvars):
    """Creates a scatterplot of the variables specified in plotvars (by their 
    indices). 
    data: array of values. size: #Events*#Parameters.
    predictions: array of class labels (here: only 0,1) size: #Events*1
    plotvars: list of variable indices to be plotted, that correspond to data.
                length: 2 """
                
    featureOrder = ['FSC-A','SSC-A','CD20','CD10','CD45','CD34','SYTO 41','CD19','CD38']
    d = np.concatenate((data[:, [plotvars[0],plotvars[1]]], predictions), axis = 1)
    c = [featureOrder[plotvars[0]], featureOrder[plotvars[1]], 'predictions']    
    df = pandas.DataFrame(data=d, columns = c)

    #scatter_matrix(df)
    sns.pairplot(df, markers = '.', hue='predictions', vars = [featureOrder[plotvars[0]], featureOrder[plotvars[1]]], size = 5)
    
    #figure1 = plt.scatter(data[plotvars[0]], data[plotvars[1]], c=predictions, cmap = 'bwr')    
    #figure1.show()
    return
    
    
def results(testLabels, predictions):
    start_time = time.time()
    
    nEvents = testLabels.shape[0]
    
    P = 0
    N = nEvents - P
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    for y, y_ in zip(testLabels, predictions):
        if y == y_ == 1:
            TP += 1            
            P += 1
        elif y == 0 and y_ == 1:
            FP += 1 
        elif y == 1 and y_ == 0:
            FN += 1
            P += 1
        elif y == y_ == 0:
            TN += 1

    TPR = TP/P
    FPR = FP/N
    FNR = FN/P
    TNR = TN/N
    
    
    ACC = (TP + TN)/nEvents
    PREC = TP/(TP + FP)
    
    
    LR_plus = TPR/FPR
    LR_minus = FNR/TNR
    ACC = (TP + TN)/nEvents
    PREC = TP/(TP + FP)
    NPV = TN/(TN + FN)
    
    predictedMRD = (TP + FP)/nEvents
    trueMRD = (TP + FN)/nEvents
    
    #function to be minimized by hyperopt 
    hyperopt_loss = FPR + FNR
    
    #cross entropy loss for the test set
    generalization_error = -np.sum(testLabels*np.log(predictions))
    
    
    cmat = []
    cmat.append("Confusion Matrix - nEvents = %d" % nEvents)
    cmat.append("TP: {0:7}, TPR: {1:.3} \t|\t FN: {2:7}, FNR: {3:.3}".format(TP, TPR, FN, FNR))    
    cmat.append("FP: {0:7}, FPR: {1:.3} \t|\t TN: {2:7}, TNR: {3:.3}".format(FP, FPR, TN, TNR))    
    
    cmat.append("\nSens. (TPR): \t {0:.3},  Spec. (TNR): \t{1:.3}".format(TPR, TNR))
    cmat.append("Precision (PPV): \t {0:.3}, Accuracy: \t {1:.3}, NPV: \t{2:.3}".format(PREC, ACC, NPV))
    cmat.append("LR+: {0:.3}, \t LR-: {1:.3}".format(LR_plus, LR_minus))
    
    duration = time.time() - start_time 
    print("calculate confusion matrix: %.3f sec" % duration) 

            
    dicResults = {
        'cmat': cmat,
        'predictedMRD': predictedMRD,
        'trueMRD': trueMRD,
        'FPR': FPR,
        'FNR': FNR,
        'loss': hyperopt_loss,
        'true_loss': generalization_error,
        'status': STATUS_OK
    }
    
    return dicResults
    
def save_log(results, continue_file=None, print_stdout=True):
    summary_vars=[
        ('TRAIN', TRAIN), 
        ('nTestSet', __n_testSet),
        ('nAeHiddenLayers', __n_ae_hidden_layers),
        ('lAeHiddenSizes', __l_ae_hidden_sizes),
        ('nMlpHiddenLayers', __n_mlp_hidden_layers),
        ('lMlpHiddenSizes', __l_mlp_hidden_sizes),
        ('dAeLearningRate', __d_ae_learning_rate),
        ('dMlpLearningRate', __d_mlp_learning_rate),
        ('nBatchSize', __n_batch_size),
        ('nMaxSteps', __n_max_steps)
    ]   
    
    
    lines = []
    lines.append(time.ctime())
    lines.append('\nParameters:')
    
    for tup in summary_vars:
        lines.append(tup[0] + ' = ' + str(tup[1]))
            
    lines.append('\nResults:')
    lines = lines + results    
    lines.append('\n' + 72*'-' + '\n')

    
    
    if continue_file == None:
        filename = __s_logdir + time.strftime('%d%m%y_%H%M%S') + '.txt', 'w'
    else:
        filename = continue_file
    
    txtfile = open(filename, 'a')
    txtfile.write('\n'.join(lines) + '\n')
    txtfile.close
    
    if print_stdout is True:
        print("\n\n--------Summary--------\n")
        for line in lines: print(line) 
            
    return
    
def run(args=None):
    """
    args: Dictionary of args. For possible arguments, see definition of parse_args(args)
    """
    
    #variables used by other functions that shouldn't be passed as an argument
    global n_events, n_features, summary_writer, summary_op 
    global x, y, ae_train_op, mlp_train_op, ae_loss, mlp_loss, output_tensor
    global trainData, trainLabels, testData, testLabels
    global global_step, index_shuf
    
    parse_args(args)
    
    #empty logdir. Tensorboard reads the whole dorectory at once, which 
    #can cause problems if the graph has changed between different runs
    
    for f in os.listdir(__s_logdir):
        os.remove(__s_logdir +  f)    
    
    tf.reset_default_graph()
    
    #load
    start_time = time.time()
    trainData, trainLabels, testData, testLabels = load(__n_testSet)
    n_events, n_features = trainData.shape 
    duration = time.time() - start_time
    print("load (overall): %.3f sec" % duration)
    
    
    #used by fill_feed_dict() to retrieve the next batch
    index_shuf = list(range(n_events))
    np.random.shuffle(index_shuf)
    

    #placeholder
    x = tf.placeholder("float", name='x-input')
    y = tf.placeholder("float", name='y-input')
    #x = tf.placeholder("float", shape=[__n_batch_size, n_features], name='x-input')
    #y = tf.placeholder("float", shape=[__n_batch_size], name='y-input')    
    
    ae_loss, ae_train_op, mlp_loss, mlp_train_op, output_tensor = build_network(x,y)
    
    summary_writer = tf.summary.FileWriter(__s_logdir)
    summary_op = tf.summary.merge_all()
    global_step = 0
    
    test_predictions = train_and_test()
        
    #plotvars = [3, 4] #test: CD10 vs CD45
    #plot_predictions(testData, predicted_labels, plotvars)
    dicResults = results(testLabels, test_predictions)
    save_log(dicResults['cmat'], "results.txt", print_stdout=True)
    
    return dicResults
    


if __name__ == "__main__":
    dicResults = run()
    
    
