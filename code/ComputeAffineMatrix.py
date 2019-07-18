# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:56:30 2017

@author: HGY
"""

import sys
import numpy as np
from scipy.io import loadmat

#%% ComputeAffineMatrix Function
def ComputeAffineMatrix(Pt1, Pt2):
    '''
    ComputeAffineMatrix:
        Computes the transformation matrix that transforms a point from
        coordinate frame 1 to coordinate frame 2
    Input:
        Pt1: N * 2 matrix, each row is a point in image 1 
            (N must be at least 3)
        Pt2: N * 2 matrix, each row is the point in image 2 that 
            matches the same point in image 1 (N should be more than 3)
    Output:
        H: 3 * 3 affine transformation matrix, 
            such that H*pt1(i,:) = pt2(i,:)
    '''

    # Check input 
    N = len(Pt1)
    if len(Pt1) != len(Pt2):
        sys.exit('Dimensions unmatched.')
    elif N<3:
        sys.exit('At least 3 points are required.')
    
    
    # Convert the input points to homogeneous coordintes.
    P1 = np.concatenate([Pt1.T, np.ones([1,N])], axis=0)
    P2 = np.concatenate([Pt2.T, np.ones([1,N])], axis=0)
    
    
    # Now, we must solve for the unknown H that satisfies H*P1=P2
    # But PYTHON needs a system in the form Ax=b, and A\b solves for x.
    # In other words, the unknown matrix must be on the right.
    # But we can use the properties of matrix transpose to get something
    # in that form. Just take the transpose of both sides of our equation
    # above, to yield P1'*H'=P2'. Then PYTHON can solve for H', and we can
    # transpose the result to produce H.
    # ref: https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
    
    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    
    # transpose of whole equation, that is, turns H*P1=P2 into P1_T*H_T=P2_T
    P1_T = np.transpose(P1)
    P2_T = np.transpose(P2)
    # Return the least-squares solution to a linear matrix equation
    H_T = np.linalg.lstsq(P1_T, P2_T)[0]
    H = np.transpose(H_T)
    
    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################  
    
    # Sometimes numerical issues cause least-squares to produce a bottom
    # row which is not exactly [0 0 1], which confuses some of the later
    # code. So we'll ensure the bottom row is exactly [0 0 1].
    H[2,:] = [0, 0, 1]

    return H
