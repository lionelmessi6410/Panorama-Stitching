# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:49:36 2017

@author: HGY
"""
import sys
import numpy as np
from ComputeAffineMatrix import ComputeAffineMatrix

#%% 
def RANSACFit(p1, p2, match, maxIter=200, seedSetSize=None, maxInlierError=30, goodFitThresh=None):
    '''
    RANSACFit Use RANSAC to find a robust affine transformation
    Input:
    p1: N1 * 2 matrix, each row is a point
    p2: N2 * 2 matrix, each row is a point
    match: M * 2 matrix, each row represents a match [index of p1, index of p2]
    maxIter: the number of iterations RANSAC will run
    seedNum: The number of randomly-chosen seed points that we'll use to fit
    our initial circle
    maxInlierError: A match not in the seed set is considered an inlier if
                its error is less than maxInlierError. Error is
                measured as sum of Euclidean distance between transformed 
                point1 and point2. You need to implement the
                ComputeCost function.
                
    goodFitThresh: The threshold for deciding whether or not a model is
               good; for a model to be good, at least goodFitThresh
               non-seed points must be declared inliers.
               
    Output:
        H: a robust estimation of affine transformation from p1 to p2
    '''

    # Check parameters
    N = len(match)
    if N<3:
        sys.exit('not enough matches to produce a transformation matrix')
    if not seedSetSize:
        seedSetSize = np.max([np.ceil(0.2*N), 3])
    if not goodFitThresh:
        goodFitThresh = np.floor(0.7*N)


    '''
    below is an obfuscated version of RANSAC. You don't need to
    edit any of this code, just the ComputeError() function below
    '''
    H = np.eye(3);
    iota = np.inf
    kappa = 0
    lamb = iota
    alpha = int(seedSetSize)
    
    for i in range(maxIter):
        # beta: randomly select alpha match points
        # gamma: the rest of match points(not in the set)
        [beta, gamma] = part(match, alpha)
        eta = ComputeAffineMatrix(p1[beta[:,0],:], p2[beta[:,1],:])
        delta = ComputeError(eta, p1, p2, gamma)
        epsilon = delta <= maxInlierError

        if np.sum(epsilon) + alpha >= goodFitThresh:
            zeta = np.concatenate([beta, gamma[epsilon, :]], axis=0)

            eta = ComputeAffineMatrix(p1[zeta[:,0], :], p2[zeta[:,1], :])
            theta = np.sum(ComputeError(eta, p1, p2, zeta))
            if theta < iota:
                H = eta
                kappa = lamb
                iota = theta
                
    if np.sum(np.square(H - np.eye(3))) == 0:
        print('No RANSAC fit was found.')

    return H

#%%
def ComputeError(H, pt1, pt2, match):
    '''
    Compute the error using transformation matrix H to 
    transform the point in pt1 to its matching point in pt2.
    Input:
        H: 3 x 3 transformation matrix where H * [x; y; 1] transforms the point
            (x, y) from the coordinate system of pt1 to the coordinate system of pt2.
        pt1: N1 x 2 matrix where each ROW is a data point [x_i, y_i]
        pt2: N2 x 2 matrix where each ROW is a data point [x_i, y_i]
        match: M x 2 matrix, each row represents a match [index of pt1, index of pt2]
        
    Output:
        dists: An M x 1 vector where dists(i) is the error of fitting the i-th 
        match to the given transformation matrix.
        Error is measured as the Euclidean distance between (transformed pt1) 
        and pt2 in homogeneous coordinates.
    '''

    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    # hint: If you have an array of indices, MATLAB can directly use it to
    # index into another array. For example, pt1(match(:, 1),:) returns a
    # matrix whose first row is pt1(match(1,1),:), second row is 
    # pt1(match(2,1),:), etc. (You may use 'for' loops if this is too
    # confusing, but understanding it will make your code simple and fast.)
    
    # select the match points
    pt1_match = pt1[match[:, 0], :]
    pt2_match = pt2[match[:, 1], :]
    N = match.shape[0]
    
    # Convert the points to homogeneous coordintes.
    pt1_homo = np.concatenate([pt1_match.T, np.ones([1,N])], axis=0)
    pt2_homo = np.concatenate([pt2_match.T, np.ones([1,N])], axis=0)
    
    # transformed pt1
    pt1_trans = np.dot(H, pt1_homo)
    
    error = pt1_trans[:2, :] - pt2_homo[:2, :]
    L2_norm = np.sqrt(np.sum(error*error, axis=0))
    dists = L2_norm.reshape((-1, 1))
    dists = dists.flatten()
    
    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################   
    
    if len(dists) != len(match):
        sys.exit('wrong format')
    

    return dists


#%% Seperate the data
def part(D, splitSize):
    randIdx = np.random.permutation(len(D))
    D1 = D[randIdx[0:splitSize],:]
    D2 = D[randIdx[splitSize:],:]
    return D1, D2
