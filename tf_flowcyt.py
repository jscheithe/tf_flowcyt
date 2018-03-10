# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:05:11 2016

@author: jakob
"""

import numpy as np
import tensorflow as tf
import math
import random
import time
import sys, os, shutil
import argparse
import matplotlib
matplotlib.use('agg') #required to run matplotlib on a machine without an X-server
import matplotlib.pyplot as plt
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

__plot = True
__to_file = True
__verbose_summaries = False

"""REMOVETest Set Number"""
__n_testSet = 0

"""Hyperparameters"""
#__n_autoencoders = 0
#__l_ae_sizes = []
#__n_autoencoders = 1
#__l_ae_sizes = [10]
#__n_autoencoders = 2
#__l_ae_sizes = [50, 15]
__n_autoencoders = 3
__l_ae_sizes = [29, 10, 11]
#__n_autoencoders = 3
#__l_ae_sizes = [200, 40, 15]
#__n_autoencoders = 4
#__l_ae_sizes = [1000, 500, 250, 30]
#

__n_mlp_hidden_layers = 1
__l_mlp_hidden_sizes = [6]
#__n_mlp_hidden_layers = 2
#__l_mlp_hidden_sizes = [15, 15]
#__n_mlp_hidden_layers = 4
#__l_mlp_hidden_sizes = [1000,500,250,30]

#__activation_function = tf.nn.sigmoid
__activation_function = tf.nn.relu
#__ae_regularization = tf.nn.l2_loss
__ae_regularization = lambda x: 0
__d_reg_beta = 0.001


def add_gaussian_noise(tensor, stddev):
    return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=stddev, dtype=tf.float32) 

__noise_adding_function = lambda x, y: x
#__noise_adding_function = add_gaussian_noise
# stddev: (0, 0.05, 0.10, 0.15, 0.30, 0.50)
#__noise_adding_function = tf.nn.dropout
__d_noise_param = 0.002

__n_classes = 2
__n_features = 9

__d_ae_learning_rate = 0.1
__ae_optimizer = tf.train.AdamOptimizer(__d_ae_learning_rate, epsilon=1)
__d_mlp_learning_rate = 0.01
__mlp_optimizer = tf.train.AdamOptimizer(__d_mlp_learning_rate, epsilon=1)
__d_min_convrate = 0.03

__n_batch_size = 1024
__n_max_steps = 24

__k = 10


"""Paths"""
__s_datadir = 'data/'
__s_csvpath = 'exp_list_PR.csv'
__s_logdir = 'log/summaries'
__s_model_save_path = 'log/models/'



def parse_args(args=None):
    """ args: dictionary of arguments (hyperparameters,...)"""   
    global summary_args, TRAIN, __n_testSet, __n_autoencoders, __l_ae_sizes, __n_mlp_hidden_layers, __l_mlp_hidden_sizes, __d_ae_learning_rate, __d_mlp_learning_rate, __n_batch_size, __n_max_steps, __d_noise_param, __d_reg_beta

    """If no parameters are passed as an argument, look for command line parameters."""
    if __name__ == "__main__" and args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('TRAIN', type=str, nargs='?', default=None,
                            help='train the model or load the last trained model from disk?')
        parser.add_argument('nTestSet', type=int, nargs='?', default=None,
                            help='number of the test set (for cross validation)')
        parser.add_argument('nAutoencoders', type=int, nargs='?', default=None,
                            help='number of autoencoders in the stacked autoencoder')
        parser.add_argument('nAeSizes', type=lambda s: [int(n) for n in s.split(',')], nargs='?', default=None,
                            help='number of nodes in the AE encoder layers, separated by commas: \',\'')
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
        parser.add_argument('dRegBeta', type=float, nargs='?', default=None,
                            help='beta (regularization magnitude)')
        parser.add_argument('dNoiseKeepProb', type=float, nargs='?', default=None,
                            help='probability that no noise is applied for a node')
        argsNamespace = parser.parse_args()
        args = vars(argsNamespace)


    if args is not None:
#            if args.get('TRAIN') is not None:    
#                TRAIN = (args['TRAIN'] == 'True')
            if args.get('nTestSet') is not None:
                __n_testSet = args['nTestSet'] 
            if args.get('nAutoencoders') is not None:
                __n_autoencoders = args['nAutoencoders']
            if args.get('nAeSizes') is not None:
                __l_ae_sizes = args['nAeSizes']
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
            if args.get('dRegBeta') is not None:
                __d_reg_beta = args['dRegBeta']
            if args.get('dNoiseKeepProb') is not None:
                __d_noise_param = args['dNoiseKeepProb']
       

def load(nTestSetNumber):
    global trainData, trainLabels, l_testData, l_testLabels 
    
    l_testData = []
    l_testLabels = []
    
    start_time = time.time()
    trainFiles, testFiles = helpers.createTrainTestH5(__s_csvpath, nTestSetNumber)
    duration = time.time() - start_time
    print("create/check h5 files: %.3f sec" % duration) 
    
    start_time = time.time()
    trainData, trainLabels  = helpers.read_h5(trainFiles)
    duration = time.time() - start_time
    print("load train data: %.3f sec" % duration)   
    
    start_time = time.time()
    for file in testFiles:
        data, labels = helpers.read_h5([file])
        l_testData.append(data)
        l_testLabels.append(labels)
        duration = time.time() - start_time
    print("load test data: %.3f sec" % duration)    
    
    return trainData, trainLabels, l_testData, l_testLabels  
    
    
def get_mse_loss(x1, x2):
    with tf.name_scope('mse'):
#        mse_loss = -tf.reduce_mean(x1 * tf.log(x2))
#        mse_loss =   -tf.reduce_mean(x2 * -tf.log(x1) + (1 - x2) * -tf.log(1 - x1))

        mse_loss = tf.reduce_mean(0.5*tf.square(tf.subtract(x1, x2)), name='mean_square_error')
        tf.summary.scalar('average_' + mse_loss.name, tf.reduce_mean(mse_loss))
    return mse_loss
        

def get_xentropy_loss(logits, labels):
    """Returns the loss tensor for the mlp network. 
    logits: batch output from the mlp network
    labels: 1-D tensor containing the labels of the current batch 
            (length: __n_batch_size)"""
    with tf.name_scope('xentropy'):
        labels = tf.cast(labels, tf.int64)
        xentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
        tf.summary.scalar('average_' + xentropy_loss.name, tf.reduce_mean(xentropy_loss))
    return xentropy_loss


def get_regularization_term(tensor):
    with tf.name_scope('regularization'):
        regterm = __d_reg_beta*__ae_regularization(tensor)
    return regterm
    
    
def add_noise(tensor):
    with tf.name_scope('noise'):
        out = __noise_adding_function(tensor, __d_noise_param)
    return out

 
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


def write_summaries(sess):
#    batches = math.floor(n_trainEvents/__n_batch_size)
#    for batch_number in range(batches):                       
#        feed_dict = fill_feed_dict(batch_number, trainData, trainLabels)
#        summary_str = sess.run(summary_op, feed_dict=feed_dict)
#        summary_writer.add_summary(summary_str, global_step)
#    summary_writer.flush()  
    #trying: only summaries for one batch
    feed_dict = fill_feed_dict(0, trainData, trainLabels, shuffle=False)
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, global_step)
    summary_writer.flush()       

def get_bias_variable(shape):
    return tf.get_variable('bias', shape, 
                           initializer=tf.constant_initializer(0.1))
    
    
def get_weights_variable(shape):
    initializer=tf.random_normal_initializer(0, 1.0 / math.sqrt(float(__n_features)))
    return tf.get_variable('weights', shape, initializer=initializer)


def build_nn_layer(input_tensor, input_dim, output_dim, layer_name, act=__activation_function):
    """Note: for the addition, the bias value is broadcast to fit the batch size.
    """
    with tf.variable_scope('net'):
        bias = get_bias_variable([output_dim])
        weights = get_weights_variable([input_dim, output_dim])
        variable_summaries(weights, layer_name + '/weights')
        net = tf.matmul(input_tensor, weights) + bias
        
    if act is None:
        return net, weights, bias
    with tf.variable_scope('activation'):
        activation = act(net)
    return activation, weights, bias
    
    
def build_sae(input_tensor, n_autoencoders, ae_sizes):
    """for the i-th autoencoder, noise is added to the input of the encoding 
    layer if the global variable lg_addNoiseToLayer[i] == True.
    lg_addNoiseToLayer is set at training time to add noise only to the input 
    of the autoencoder to be trained and pass the uncorrupted input through all
    the other encoder layers"""    
    
    if n_autoencoders == 0:
            return input_tensor, None, None
    
    global lb_addNoiseToLayer
#    lb_addNoiseToLayer = [tf.cast(False, tf.bool) for i in range(n_autoencoders)]    

    noise_list = [0]*n_autoencoders
#    lb_addNoiseToLayer = tf.Variable(noise_list, trainable=False, name='noise-list')
    
    enc_activation = []
    enc_weights = []
    enc_bias = []
    
    dec_activation = []
    
    ae_loss_func = []
    ae_train_op = []
        
    with tf.variable_scope('sae'):
        lb_addNoiseToLayer = tf.Variable(noise_list, trainable=False, name='noise-list')
        with tf.variable_scope('ae1'):
            with tf.variable_scope('encoder'):
                uncorrupted_input = input_tensor          
                with tf.variable_scope('training-noise'):
                    corrupted_input = tf.cond(tf.equal(lb_addNoiseToLayer[0], 1), lambda: add_noise(uncorrupted_input), lambda: uncorrupted_input)
                act1, w1, b1 = build_nn_layer(corrupted_input, __n_features, ae_sizes[0], 'ae1_encoder')
                enc_activation.append(act1)
                enc_weights.append(w1)
                enc_bias.append(b1)
            with tf.variable_scope('decoder'):
                act2, w2, b2 = build_nn_layer(enc_activation[0], ae_sizes[0], __n_features, 'ae1_decoder')
                dec_activation.append(act2)
                
            #only train the first autoencoder        
            with tf.variable_scope('loss'):
                loss = tf.add(get_mse_loss(uncorrupted_input, dec_activation[0]), get_regularization_term(w1) + get_regularization_term(w2), name='regularized-mse')
                ae_loss_func.append(loss)
            with tf.variable_scope('AdamOptimizer'):
                var_list = [w1, b1, w2, b2]                
                ae_train_op.append(__ae_optimizer.minimize(ae_loss_func[0], var_list = var_list))
     
        for i in range(1, n_autoencoders):
            with tf.variable_scope('ae' + str(i+1)):
                with tf.variable_scope('encoder'):
                    uncorrupted_input = enc_activation[i-1]        
                    with tf.variable_scope('training-noise'):
                        corrupted_input = tf.cond(tf.equal(lb_addNoiseToLayer[i], 1), lambda: add_noise(uncorrupted_input), lambda: uncorrupted_input)
                    act1, w1, b1 = build_nn_layer(corrupted_input, ae_sizes[i-1], ae_sizes[i], 'ae' + str(i+1) + '_encoder')
                    enc_activation.append(act1)
                    enc_weights.append(w1)
                    enc_bias.append(b1)
                with tf.variable_scope('decoder'):
                    act2, w2, b2 = build_nn_layer(enc_activation[i], ae_sizes[i], ae_sizes[i-1], 'ae' + str(i+1) + '_decoder')
                    dec_activation.append(act2)   
                    
                #only train the i-th autoencoder
                with tf.variable_scope('loss'):
                    loss = tf.add(get_mse_loss(uncorrupted_input, dec_activation[i]),  get_regularization_term(w1) + get_regularization_term(w2), name='regularized-mse')
                    ae_loss_func.append(loss)
                with tf.variable_scope('AdamOptimizer'):
                    var_list = [w1, b1, w2, b2]
                    ae_train_op.append(__ae_optimizer.minimize(ae_loss_func[i], var_list = var_list))
        
        if __verbose_summaries:
            with tf.name_scope('evaluation'):
                with tf.name_scope('dropout'):
                    for i in range(n_autoencoders):
                        tf.summary.scalar('dropout-layer-' + str(i), lb_addNoiseToLayer[i])
                with tf.name_scope('encoder_stack_activation'):                
                    activation = enc_activation[-1]
                    tf.summary.tensor_summary('encoder_stack_activation', activation)
                    for i in range(5):
                        tf.summary.scalar('event0:encoder_stack_activation_node' + str(i), activation[0,i])
                        tf.summary.scalar('event1:encoder_stack_activation_node' + str(i), activation[1,i])
    
    sae_output = enc_activation[-1]
    return sae_output, ae_train_op, ae_loss_func


def build_feedforward_network(input_tensor, hidden_layers, hidden_sizes, type='mlp'):
    """Builds an mlp network of arbitrary shape.
    input_tensor: here, the evaluated output of the ae network (tensor of shape [batch_size, __n_features])
    The first layer recieves the ae network's output.    
    The hidden layers use sigmoid activation function, the last layer (for classification)
    gives unscaled output. Softmax activation is done within the loss function.
    todo: for summaries, tags are made unique by prepending the string type. This
          could be a problem if more than one network of the same type is used."""
    
    if __n_autoencoders == 0:
        n_inputDim = __n_features
    else:
        n_inputDim = input_tensor.get_shape().as_list()[1]
        
    activations = []
    weights = []
    biases = []
    for i in range(hidden_layers):  #we build the hidden layers
        with tf.variable_scope('layer' + str(i)):                
            full_layer_name = type + '_layer' + str(i)
            if i == 0:
                new_layer, w, b = build_nn_layer(input_tensor, n_inputDim, hidden_sizes[i], full_layer_name) 
                activations.append(new_layer)
                continue                
            new_layer, w, b = build_nn_layer(activations[i-1], hidden_sizes[i-1], hidden_sizes[i], full_layer_name)                
            activations.append(new_layer)
            weights.append(w)
            biases.append(b)
    with tf.variable_scope('output'):
        #for an mlp, the output layer consists of __n_classes nodes and doesn't use an activation function (activation is calculated within the loss function)
        output, w, b = build_nn_layer(activations[-1], hidden_sizes[-1], __n_classes, 'mlp/output_layer', act=None)
        weights.append(w)
        biases.append(b)
    return output, weights, biases


def build_mlp(input_tensor, n_hiddenlayers, l_hidden_sizes):
    with tf.variable_scope('mlp'):
        with tf.variable_scope('network'):
            mlp_output, weights, biases = build_feedforward_network(input_tensor, n_hiddenlayers, l_hidden_sizes)
            with tf.variable_scope('loss'):                
                mlp_loss = get_xentropy_loss(mlp_output, y)  
#            mlp_var_list = weights + biases
            mlp_var_list = tf.trainable_variables()
            with tf.variable_scope('AdamOptimizer'):
                mlp_train_op = __mlp_optimizer.minimize(mlp_loss, var_list = mlp_var_list)
        
        with tf.name_scope('evaluation'):
            with tf.name_scope('softmax_activation'):                
                softmax_output = tf.nn.softmax(mlp_output)
#                for i in range(10):
#                    tf.summary.scalar('softmax_output_class1' + str(i), softmax_output[i,1])
#            with tf.name_scope('labels'):
#                for i in range(10):
#                    tf.summary.scalar('label' + str(i), y[i])
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(y, tf.cast(tf.argmax(softmax_output, 1), tf.float32))
#                for i in range(10):
#                    tf.summary.scalar('correct_prediction' + str(i), tf.cast(correct_prediction[i], tf.int64))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
                tf.summary.scalar('accuracy', accuracy)
    return mlp_output, mlp_train_op, mlp_loss
   
    
def fill_feed_dict(n_batchnumber, data, labels, shuffle=True):
    """Fills the feed_dict with a random batch of from trainData.
    Uses the shuffeled list of indices index_shuf to randomly draw training 
    data. Every value is only drawn once, as n_batchnumber runs from 0 to 
    n_trainEvents/__n_batch_size
    n_batchnumer < 0: fill feed dict with the whole set of training data (unshuffeled)."""
    
    if n_batchnumber < 0:
        labels_reshape = np.reshape(labels, np.shape(labels)[0])        
        feed_dict = {
            x: data,
            y: labels_reshape
        }
        return feed_dict
    
    n_start_index = n_batchnumber*__n_batch_size
    batch_data = np.zeros([__n_batch_size, __n_features])
    batch_labels = np.zeros([__n_batch_size])
    for i in range(__n_batch_size):    
        if shuffle:
            batch_data[i,:] = data[index_shuf[n_start_index + i], :]
            batch_labels[i] = labels[index_shuf[n_start_index + i]]
        else:
            batch_data[i,:] = data[n_start_index + i, :]
            batch_labels[i] = labels[n_start_index + i]
    
    feed_dict = {
        x: batch_data,
        y: batch_labels
    }    
    return feed_dict    


def do_training(sess, train_op, loss):
    
    write_summaries(sess)

    global global_step
    global index_shuf
    batches = math.floor(n_trainEvents/__n_batch_size)
    last_loss = np.inf
    for step in range(__n_max_steps):
        start_time = time.time()
        loss_value = 0.0
        np.random.shuffle(index_shuf)
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
            print('%s: Step %d: average: %.12f (%.3f sec)' % (loss.name, step, loss_value, duration))
            #with tf.variable_scope('mlp', reuse=True):
#                with tf.variable_scope('Evaluation', reuse=True):
#                    with tf.variable_scope('accuracy', reuse=True):
#                        accuracy = tf.get_variable('accuracy')
#                        tf.scalar_summary('mlp/Evaluation/accuracy/accuracy', accuracy)
            """uncomment for summaries for every step"""
            if __verbose_summaries:            
                write_summaries(sess)
        
        global_step += 1  

        convergence_rate = loss_value/last_loss
#        print(convergence_rate)
        if(abs(convergence_rate - 1)) < __d_min_convrate:
            break

        last_loss = loss_value
    return loss_value
    
    
def train_sae(sess, ae_train_op, ae_loss_func):
    global lb_addNoiseToLayer
    
    assert len(ae_train_op) == len(ae_loss_func)
    nLayers = len(ae_train_op)
    
    for i in range(nLayers):        
        val = lb_addNoiseToLayer.assign([0]*i + [1] + [0]*(nLayers - (i + 1)))
        sess.run(val)
        do_training(sess, ae_train_op[i], ae_loss_func[i])
    val = lb_addNoiseToLayer.assign([0]*nLayers)
    sess.run(val)

    
    
def train_model(sess, ae_train_op, ae_loss, mlp_train_op, mlp_loss):
    saver = tf.train.Saver()        
    if TRAIN == True:
        start_time = start_time_train = time.time()
        if ae_train_op is not None:
            train_sae(sess, ae_train_op, ae_loss)
            duration = time.time() - start_time
            print("AE training: %.3f sec" % duration)
        
        start_time = time.time()
        do_training(sess, mlp_train_op, mlp_loss)
        duration = time.time() - start_time
        print("MLP training: %.3f sec" % duration)
        
        duration_train = time.time() - start_time_train
        print('train (overall)): %.3f' % duration_train)
        
#        saver.save(sess, __s_model_save_path + time.strftime('%Y%m%d-%H%M%S'))
#    else:
#        saver.restore(sess, tf.train.latest_checkpoint(__s_model_save_path))
#    write_summaries(sess)
    return 


def predict(sess, model_output):
    """model_output: activation of the classificator before softmax (logits).
    returns: a list of predictions, one entry for each sample (patient)"""
    global l_testData, l_testLabels     
    start_time = time.time()

    predictions_tensor = tf.nn.softmax(model_output)
    l_predictions = [np.zeros((np.shape(testData)[0], 2)) for testData in l_testData] 
    l_predicted_labels = []
        
    """Evaluating all predictions at one time is too resource expensive.
       Thus evaluating in batches."""
    
    for sampleNumber in range(len(l_testData)):
        testData = l_testData[sampleNumber]
        testLabels = l_testLabels[sampleNumber]
        
        n_test_events = np.shape(testData)[0]
        batches = math.floor(n_test_events/__n_batch_size)
#        feed_dict = fill_feed_dict(-1, testData, testLabels, shuffle=False)
#        l_predictions[sampleNumber] = sess.run(predictions_tensor, feed_dict = feed_dict)
        
        for batch_number in range(batches):    
            startindex = batch_number*__n_batch_size
            endindex = ((batch_number+1)*__n_batch_size)        
            
            feed_dict = fill_feed_dict(batch_number, testData, testLabels, shuffle=False)
            batch_predictions = sess.run(predictions_tensor, feed_dict = feed_dict)
            l_predictions[sampleNumber][startindex:endindex, :] = batch_predictions
       
        l_predicted_labels.append(np.reshape(np.argmax(l_predictions[sampleNumber],1), (-1, 1)))

#    correct_prediction = np.equal(testLabels, predicted_labels)
#    accuracy = sum(correct_prediction)/len(correct_prediction)
#    print('test accuracy: %.3f' % (accuracy))
    
    duration = time.time() - start_time
    print("predict: %.3f sec" % duration)
    
    return l_predicted_labels
    
    

    
def results(l_testLabels, l_predictions):
    start_time = time.time()

# uncomment to calculate the values jointly for the whole dataset
#    testLabels = [label for labels in l_testLabels for label in labels]
#    predictions = [pred for predictions in l_predictions for pred in predictions]    
    assert len(l_testLabels) == len(l_predictions)   
    nSamples = len(l_testLabels)
    nEvents = 0
    
    TP = 0 
    FP = 0
    TN = 0
    FN = 0
    TPR = 0
    FPR = 0
    TNR = 0
    FNR = 0
    ACC = 0
    PREC = 0    
    LR_plus = 0
    LR_minus = 0
    NPV = 0
    predictedMRDs = np.zeros(nSamples)
    trueMRDs = np.zeros(nSamples)

    
    for i in range(nSamples):
        testLabels = l_testLabels[i]
        predictions = l_predictions[i]
        
        n = len(testLabels)
        tp = fp = tn = fn = 0
        
        for y, y_ in zip(testLabels, predictions):
            if y == y_ == 1:
                tp += 1            
            elif y == 0 and y_ == 1:
                fp += 1 
            elif y == 1 and y_ == 0:
                fn += 1
            elif y == y_ == 0:
                tn += 1
        pmrd = (tp + fp)/n
        tmrd = (tp + fn)/n
        predictedMRDs[i] = pmrd
        trueMRDs[i] = tmrd
    
        nEvents += n
        TP += tp
        FP += fp
        TN += tn
        FN += fn
    
    P = TP + FN
    N = TN + FP
    
    TPR = TP/np.float64(P)
    FPR = FP/np.float64(N)
    FNR = FN/np.float64(P)
    TNR = TN/np.float64(N)
    
    ACC = (TP + TN)/np.float64(nEvents)
    PREC = TP/np.float64((TP + FP))
    
    F1 = 2*(PREC*TPR)/np.float64(PREC + TPR)
    
    LR_plus = TPR/np.float64(FPR)
    LR_minus = FNR/np.float64(TNR)
    NPV = TN/np.float64((TN + FN))
    
#    mrdMAPE = (np.sum(np.abs((trueMRDs - predictedMRDs)/trueMRDs)))/nEvents
#    mrdMAPE = mrdMAPE[mrdMAPE<1e10]     #remove inf entries (where trueMRD was 0)
    
    mrdRes = np.array(trueMRDs) - np.array(predictedMRDs)
        
#    hyperopt_loss = mrdMAPE
#    hyperopt_loss = max(0, min(2, 2 - (ACC + PREC)))    #avoid inf
    if np.isnan(F1) or np.isinf(F1):
        F1 = 0.0
    hyperopt_loss = 1 - F1
    #cross entropy loss for the test set
    #generalization_error = -np.sum(testLabels*np.log(predictions))
        
    
    cmat = []
    cmat.append("# of test events: %d, \n# of samples: %d, \n# of events per sample (average): %f \nBLAST rate: %f" % (nEvents, nSamples, nEvents/np.float64(nSamples), P/np.float64(nEvents)))
    cmat.append("\n------------------------ Confusion Matrix ------------------------")
    cmat.append("TP: {0:7}, TPR: {1:.3}  |  FN: {2:7}, FNR: {3:.3}".format(TP, TPR, FN, FNR))    
    cmat.append("FP: {0:7}, FPR: {1:.3}  |  TN: {2:7}, TNR: {3:.3}".format(FP, FPR, TN, TNR))    
    cmat.append("\nSens. (TPR):    \t {0:.3}, \tSpec. (TNR): \t{1:.3}".format(TPR, TNR))
    cmat.append("Precision (PPV): \t {0:.3}, \tAccuracy:    \t{1:.3}".format(PREC, ACC))
    cmat.append("\nLR+: {0:.3},   LR-: {1:.3},   NPV: {2:.3},   F1: {3:.3}".format(LR_plus, LR_minus, NPV, F1))
    cmat.append("\n------- Residuals (true MRD - predicted MRD) -------")
    cmat.append(str(mrdRes))
    cmat.append("Mean sq. error: %f" %  np.mean(mrdRes**2))
#    cmat.append("MRD mean absolute percentage error (MAPE): %f" %  mrdMAPE)
    
    duration = time.time() - start_time 
    print("calculate confusion matrix: %.3f sec" % duration) 
            
    dicResults = {
        'cmat' : cmat,
        'duration' : 0,
        'predictedMRDs': predictedMRDs,
        'trueMRDs': trueMRDs,
#        'mrdMAPE' : mrdMAPE,
        'F1' : F1,
        'FPR': FPR,
        'FNR': FNR,
        'ACC' : ACC,
        'PREC': PREC,      
        'loss' : hyperopt_loss,
        'status': STATUS_OK
    }
    
    return dicResults
    
def save_log(results, continue_file=None, print_stdout=True):
    summary_vars=[
        ('TRAIN', TRAIN), 
        ('nTestSet', __n_testSet),
        ('nAutoencoders', __n_autoencoders),
        ('lAeSizes', __l_ae_sizes),
        ('nMlpHiddenLayers', __n_mlp_hidden_layers),
        ('lMlpHiddenSizes', __l_mlp_hidden_sizes),
        ('dAeLearningRate', __d_ae_learning_rate),
        ('dMlpLearningRate', __d_mlp_learning_rate),
        ('nBatchSize', __n_batch_size),
        ('nMaxSteps', __n_max_steps),
        ('activationFunction', __activation_function),
        ('aeRegularization', __ae_regularization),
        ('dRegBeta', __d_reg_beta),
        ('noiseFunction', __noise_adding_function),
        ('dNoiseKeepProb', __d_noise_param),
        ('dMinConvRate', __d_min_convrate)
    ]
    
    lines = []
    lines.append(time.ctime())
    lines.append('\nParameters:\n')
    
    for tup in summary_vars:
        lines.append('\t' + tup[0] + ' = ' + str(tup[1]))
            
    lines.append('\nEvaluation:')
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

    
def k_fold_xvalidation(k, ae_train_op, ae_loss_func, mlp_output, mlp_train_op, mlp_loss_func):
    global n_trainEvents
    global x, y
    global trainData, trainLabels, l_testData, l_TestLabels
    global index_shuf
    
    l_dic_Results = [] #one dic for every k
    l_l_test_predictions = [] #one list of sample predictions for every k
    l_model_files = []
    cumul_fscore = 0

##    empty logdir. Tensorboard reads the whole dorectory at once, which 
##    can cause problems if the graph has changed between different runs
#    for f in os.listdir(__s_logdir):
#        os.remove(__s_logdir +  f)    
#    
    for i in range(k):
        print(str(k) + '-fold cross validation: k = ' + str(i+1))
                    
#        if k==1 the test set should be chosen via a command line parameter
        if k != 1:
            nTestSet = i
        else:
#            nTestSet = __n_testSet
            nTestSet = random.randint(1,10)
#        nTestSet = random.randint(1,10)
            
        #load
        start_time = time.time()
        trainData, trainLabels, l_testData, l_testLabels = load(nTestSet)
        n_trainEvents, _ = trainData.shape 
        duration = time.time() - start_time
        print("load (overall): %.3f sec" % duration)
        
        
        #used by fill_feed_dict() to retrieve the next batch
        index_shuf = list(range(n_trainEvents))
        np.random.shuffle(index_shuf)
               
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            summary_writer.add_graph(sess.graph, global_step=global_step)
            
            train_model(sess, ae_train_op, ae_loss_func, mlp_train_op, mlp_loss_func)
            l_l_test_predictions.append(predict(sess, mlp_output))
     
            # Update the events file with results from the whole data set.
#            write_summaries(sess)
        
        
        dic_Results = results(l_testLabels, l_l_test_predictions[i])

        cumul_fscore += dic_Results['F1']        
        
        l_dic_Results.append(dic_Results)

        save_log(dic_Results['cmat'], "results.txt", print_stdout=True)
        
    print('\nAverage F1-Score: {:.4}'.format(cumul_fscore/k))

    return l_dic_Results, l_l_test_predictions, l_model_files


def plot_mrd(l_dic_Results, savepath=''):
    predictedMRDs = np.empty(0)
    trueMRDs = np.empty(0)
    for dic in l_dic_Results:
        predictedMRDs = np.concatenate((predictedMRDs, dic['predictedMRDs']))
        trueMRDs = np.concatenate((trueMRDs, dic['trueMRDs']))
    
    x = np.logspace(-6, 0)
    y = np.logspace(-6, 0)
    
    fig = plt.figure(figsize=(5,5), facecolor='w', edgecolor='k', linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None)
    ax1 = fig.add_subplot(111)
    
    ax1.scatter(trueMRDs, predictedMRDs)
    ax1.plot(x,y)

    ax1.set_xlabel('trueMRD')
    ax1.set_ylabel('predictedMRD')    
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    plt.savefig(savepath + 'plot_mrd.png', dpi = 300)

    


def plot_predictions(l_data, l_l_predictions, plotvars=None, savepath=''):
    """Creates a scatterplot of the variables specified in plotvars (by their 
    indices). 
    data: array of values. size: #Events*#Parameters.
    predictions: array of class labels (here: only 0,1) size: #Events*1
    plotvars: list of variable indices to be plotted, that correspond to data.
                length: 2 """
                
    if plotvars == None:
        plotvars = [3, 4]
        
    data = l_data[0]
    predictions = l_l_predictions[-1][0] #stimmts?
    
#    data = [event for data in l_data for event in data]
#    l_predictions = [l_pred for l_predictions in l_l_predictions for l_pred in l_predictions]    
#    predictions = [pred for predictions in l_predictions for pred in predictions]    

          
    featureOrder = ['FSC-A','SSC-A','CD20','CD10','CD45','CD34','SYTO 41','CD19','CD38']
    d = np.concatenate((data[:, [plotvars[0],plotvars[1]]], predictions), axis = 1)
    c = [featureOrder[plotvars[0]], featureOrder[plotvars[1]], 'predictions']    
    df = pandas.DataFrame(data=d, columns = c)

    #scatter_matrix(df)
    pal = ['#808080', '#de0000']
    #sns.set_palette(pal)
    sns_plot = sns.pairplot(df, markers = '.', hue='predictions', palette=pal, vars = [featureOrder[plotvars[0]], featureOrder[plotvars[1]]], size = 4)
    sns_plot.savefig(savepath + 'plot_predictions' + featureOrder[plotvars[0]] + 'vs' + featureOrder[plotvars[1]] + '.png', dpi=300)

    #figure1 = plt.scatter(data[plotvars[0]], data[plotvars[1]], c=predictions, cmap = 'bwr')    
    #figure1.show()
    return
    
#def get_hyperopt_dicResults(l_dicResults):
#    """From a list of results (one entry for each sample) this function calculates
#    the "average performance" that can be used by hyperopt for hyperparameter tuning."""
#
#        'predictedMRD': predictedMRD,
#        'trueMRD': trueMRD,
#        'F1' : F1,
#        'FPR': FPR,
#        'FNR': FNR,
#        'ACC' : ACC,
#        'PREC': PREC,        
#        'status': STATUS_OK
#
#    nSamples = len(l_dicResults)    
#    
#    mrdRatios = np.zeros(nSamples)
#    F1s = np.zeros(nSamples)
#    FPR = np.zeros(nSamples)
#    FNR = np.zeros(nSamples)
#    ACC = np.zeros(nSamples)
#    PREC = np.zeros(nSamples)
#    
#       
#    for i in range(nSamples):
#        dic_Results = l_dicResults[i]
#        mrdRatios[i] = dic_Results['trueMRD']/dic_Results['predictedMRD']
#        F1s[i] = dic_Results['F1']
#
#    hyperopt_loss = np.mean(mrdRatios)        
#        
#    dicResults = {
#        'loss': hyperopt_loss,
#        'status': STATUS_OK
#    }
#    
#    return dicResults
    
def copy_logs(resultsdir):
#    for file in [f for f in os.listdir(__s_logdir) if os.path.isfile(__s_logdir + f)]:
#        tf_logfile = __s_logdir + file
#        shutil.move(tf_logfile, resultsdir + 'tf.events.out')   
        shutil.move(__s_logdir, resultsdir)   
#        os.popen('cp ' + tf_logfile + ' "' + resultsdir + 'tf.events.out"')
#        os.remove(tf_logfile)
#    for file in [f for f in os.listdir(__s_model_save_path) if os.path.isfile(__s_model_save_path + f)]:
#        tf_modelfile = __s_model_save_path + file
#        shutil.move(tf_modelfile, resultsdir + 'tf.saver_data' + os.path.splitext(tf_modelfile)[1])   
        shutil.move(__s_model_save_path, resultsdir)   
#        os.popen('cp ' + tf_modelfile + ' "' + resultsdir + 'tf.saver_data' + os.path.splitext(tf_modelfile)[1] + '"')
#        os.remove(tf_modelfile)
     
        
def run(args=None):
    """
    args: Dictionary of args. For possible arguments, see definition of parse_args(args)
    """
    start_time = time.time()    
    
    global x, y
    global summary_writer, summary_op
    global global_step

    parse_args(args)
    
    #max k: 11
    k = __k

    if not os.path.exists(__s_logdir):
        os.mkdir(__s_logdir)
    if not os.path.exists(__s_model_save_path):
        os.mkdir(__s_model_save_path)        
    
    
    s_identifier = 'tf_flowcyt.run() on ' + time.asctime()
    resultsdir = 'results/' + s_identifier + '/'
    os.mkdir(resultsdir)
    
    stdout = sys.stdout
    stderr = sys.stderr
    
    with open(resultsdir + 'tf_flowcyt.run().out', 'w') as f:
        if __to_file:        
            sys.stdout = f
            sys.stderr = f
        
        tf.reset_default_graph()
    
        #placeholder
        x = tf.placeholder("float", name='x-input')
        y = tf.placeholder("float", name='y-input')
        
        #set up the graph
        sae_output, ae_train_op, ae_loss_func = build_sae(x, __n_autoencoders, __l_ae_sizes)
        mlp_output, mlp_train_op, mlp_loss_func = build_mlp(sae_output, __n_mlp_hidden_layers, __l_mlp_hidden_sizes)
        
        summary_writer = tf.summary.FileWriter(__s_logdir)
        summary_op = tf.summary.merge_all()
        global_step = 0
        
        #training and testing happens here
        l_dic_Results, l_l_predictions, l_model_files = k_fold_xvalidation(k, ae_train_op, ae_loss_func, mlp_output, mlp_train_op, mlp_loss_func)
#
        if __plot:
            plot_mrd(l_dic_Results, savepath=resultsdir)
    #        
            plot_predictions(l_testData, l_l_predictions, [3,4], savepath=resultsdir) # CD10vsCD45
            plot_predictions(l_testData, l_l_predictions, [1,7], savepath=resultsdir) # SSCvsCD19
            plot_predictions(l_testData, l_l_predictions, [1,0], savepath=resultsdir) # SSCvsFSC
    
        copy_logs(resultsdir)
        
        duration = time.time() - start_time
        print("\noverall runtime: %.3f sec" % duration)
        
            
        dicResults = {k:  np.mean([dic[k] for dic in l_dic_Results]) for k in l_dic_Results[0] if k not in ['cmat', 'predictedMRDs', 'trueMRDs', 'status']}    
        dicResults['status'] = STATUS_OK        
        dicResults['duration'] = duration
        
        print('average results:')
        try:
            print(str(dicResults))
        except:
            pass

    if __to_file:
        sys.stdout = stdout  
        sys.stderr = stderr

    #the return value dicResults is used for hyperparameter tuning, here k should be 1 
#    dicResults = l_dic_Results[0]    


    
    return dicResults, resultsdir

if __name__ == "__main__":
    run()
    
    
