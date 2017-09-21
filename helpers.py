# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:15:26 2016

@author: jakob
"""

import os
import glob
import csv
import time

datadir = 'data/'
autoflowdir = datadir + 'autoflow/'
h5dir = datadir + 'h5/'
unbalanceddir = h5dir + 'unbalanced/'
balanceddir = h5dir + 'balanced/'

"""Fills the subfolders 'data/h5/balanced', 'data/h5/unbalanced' with the h5-files, 
created from the data in 'data/autoflow'. Training data is balanced, test data is 
unbalanced. The sample numbers for the N-th test set, as specified in the N-th 
column of the file specified in csvpath. The .bin files in 'data/autoflow' start with 
their sample number followed by an underscore. Nonexisting h5 files are generated using 
h5_creator.py."""

def createTrainTestH5(csvpath, N):    
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


        
        