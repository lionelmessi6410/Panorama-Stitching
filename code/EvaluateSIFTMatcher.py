# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:47:35 2017

@author: HGY
"""

'''
# EvaluateSIFTMatcher.py
# Run this script to test your SIFTSimpleMatcher() function
# using sample data. You do not need to change anything in this script.
'''

from scipy.io import loadmat
from SIFTSimpleMatcher import SIFTSimpleMatcher
import numpy as np

#%% Test Data (You should not change the data here)
data = loadmat('../checkpoint/Match_input.mat');
input_d1 = data['input_d1']
input_d2 = data['input_d2']
del data

#%% Call my implementation of SIFTSimpleMatcher.m
M = SIFTSimpleMatcher(input_d1, input_d2);

#%% Load data and check solution (You should not change this part.)
solution = loadmat('../checkpoint/Match_ref.mat')['solution']
print('\nYour error with the reference solution...')
print(np.sum(np.square(M-solution)))

if (np.sum(np.square(M-solution)) < 1e-30):
    print('\nAccepted!');
else:
    print('\nThere is something wrong.')
