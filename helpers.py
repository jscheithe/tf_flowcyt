# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:15:26 2016

@author: jakob
"""

import os
import subprocess
import glob
import csv
import time
import numpy as np
import h5py

datadir = 'data/'
autoflowdir = datadir + 'autoflow_all/'
h5dir = datadir + 'h5/'
unbalanceddir = h5dir + 'unbalanced/'
balanceddir = h5dir + 'balanced/'


def createTrainTestH5(csvpath, N):    
    """Fills the subfolders 'data/h5/balanced', 'data/h5/unbalanced' with the h5-files, 
    created from the data in 'data/autoflow'. Training data is balanced, test data is 
    unbalanced. The sample numbers for the N-th test set, as specified in the N-th 
    column of the file specified in csvpath. The .bin files in 'data/autoflow' start with 
    their sample number followed by an underscore. Nonexisting h5 files are generated using 
    h5_creator.py."""
    
    trainFiles = []
    testFiles = []
    lTestSamples = []       
    start_time = time.time()
    with open(csvpath, newline = '') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            lTestSamples.append(row[N])    #collect the N-th entry of each row.

    duration = time.time() - start_time
    print("read csv: %.3f sec" % duration)
    
    start_time = time.time()
    #for f in glob.glob(autoflowdir + '*.bin'):
    for f in os.listdir(autoflowdir):
        if(f.endswith('.bin')):
            sampleNumber = f.split('_')[0]
            if sampleNumber in lTestSamples:
                if glob.glob(unbalanceddir + str(sampleNumber) + '*') == []:
                    binfilepath = autoflowdir + f
                    os.system("python h5_creator.py " + binfilepath + " " + unbalanceddir + " False")
                    if glob.glob(unbalanceddir + str(sampleNumber) + '*') == []:
                        #create dummy  file
                        open(unbalanceddir + str(sampleNumber) + '_EMPTY.h5', 'a').close()
                        continue                
                nonemptyNewTestFiles = [x for x in glob.glob(unbalanceddir + str(sampleNumber) + '*') if 'EMPTY' not in x]
                testFiles  = testFiles + nonemptyNewTestFiles
            else:
                if glob.glob(balanceddir + str(sampleNumber) + '*') == []:
                    binfilepath = autoflowdir + f
                    os.system("python h5_creator.py " + binfilepath + " " + balanceddir + " True")
                    if glob.glob(balanceddir + str(sampleNumber) + '*') == []:
                        #create dummy  file
                        open(balanceddir + str(sampleNumber) + '_EMPTY.h5', 'a').close()
                        continue
                nonemptyNewTrainFiles = [x for x in glob.glob(balanceddir + str(sampleNumber) + '*') if 'EMPTY' not in x]
                trainFiles  = trainFiles + nonemptyNewTrainFiles
    
    duration = time.time() - start_time
    print("check h5 files: %.3f sec" % duration)

    return trainFiles, testFiles

def createAllH5():         
    for f in os.listdir(autoflowdir):
        if(f.endswith('.bin')):
            binfilepath = autoflowdir + f
            out1 = subprocess.check_output(["python", "tf_flowcyt/h5_creator.py", binfilepath, unbalanceddir, "False"])
            print(str(out1))
            out2 = subprocess.check_output(["python", "tf_flowcyt/h5_creator.py", binfilepath, balanceddir, "True"])
            print(str(out2))
#            os.system("python h5_creator.py " + binfilepath + " " + unbalanceddir + " False")
#            os.system("python h5_creator.py " + binfilepath + " " + balanceddir + " True")
                   
    

def read_h5(files):
    data = np.empty((0,9))
    labels = np.empty((0,1))
    for filename in files:
        with h5py.File(filename, "r") as f:
            dsetX = f['data'][()]
            dsetY = f['label'][()]   
            data = np.concatenate((data, np.array(dsetX)))
            labels = np.concatenate((labels, np.array(dsetY)))
    return data, labels     
    

