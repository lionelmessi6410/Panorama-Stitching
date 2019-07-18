# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:53:25 2017

@author: HGY
"""

'''
# EvaluateAffineMatrix.py
# Run this script to test your ComputeAffineMatrix() function
# using sample data. You do not need to change anything in this script.
'''

import numpy as np
from scipy.io import loadmat
from ComputeAffineMatrix import ComputeAffineMatrix

#%% Test Data (You should not change the data here)
srcPt = np.asarray([[0.5, 0.1], [0.4, 0.2], [0.8, 0.2]])
dstPt = np.asarray([[0.3, -0.2], [-0.4, -0.9], [0.1, 0.1]])

#%% Calls your implementation of ComputeAffineMatrix.m
H = ComputeAffineMatrix(srcPt, dstPt)

#%% Load data and check solution
solution = loadmat('../checkpoint/Affine_ref.mat')['solution']
error = np.sum(np.square(H-solution))
print('Difference from reference solution: ',str(error))

if error < 1e-20:
    print('\nAccepted!')
else:
    print('\nThere is something wrong.')