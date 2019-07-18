# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:33:20 2017

@author: HGY
"""
import numpy as np
from cyvlfeat.sift import sift
import matplotlib.pyplot as plt
from PIL import Image
from SIFTSimpleMatcher import SIFTSimpleMatcher
from RANSACFit import RANSACFit
from PairStitch import PairStitch

#%% Paths
saveFileName = '../results/uttower_pano.jpg'
img1Path = '../data/uttower1.jpg'
img2Path = '../data/uttower2.jpg'

#%% Load image
img1 = Image.open(img1Path, 'r')
img2 = Image.open(img2Path, 'r')

#%% Feature detection
def swapcolumn(arr):
    # sweep sift index of [Y,X] -> [X,Y] for easy computation of affine transform
    swap = []
    swap.append(arr[:,1])
    swap.append(arr[:,0])
    swap = np.asarray(swap).T
    return swap

'''
Extracts a set of SIFT features from 'image'. 'image' must be float32 and 
greyscale (either a single channel as the last axis, or no channel). 

sift(image, n_octaves=None, n_levels=3,  first_octave=0,  peak_thresh=0,
         edge_thresh=10, norm_thresh=None,  magnification=3, window_size=2,
         frames=None, force_orientations=False, float_descriptors=False,
         compute_descriptor=False, verbose=False)
- compute_descriptors=True, computes the SIFT descriptors as well

Output
- frames : F x 4 array
    F is the number of keypoints (frames) used. This is the center 
    of every dense SIFT descriptor that is extracted.
    Each column of frames is a feature frame and has the format [Y, X, S, TH]
    - (Y, X): the floating point center of the keypoint
    - S: the scale
    - TH: the orientation (in radians)

- descriptors : F x 128  array
    F is the number of keypoints (frames) used. The 128 length vectors 
    per keypoint extracted, uint8 by default. Only returned if compute_descriptors=True

Ref: https://github.com/menpo/cyvlfeat/blob/master/cyvlfeat/sift/sift.py
'''

I = np.asarray(img1.convert('L')).astype('single')  # rgb2gray
[f,desc1] = sift(I, compute_descriptor=True, float_descriptors=True)
pointsInImage1 = swapcolumn(f[:,0:2])

I = np.asarray(img2.convert('L')).astype('single') # rgb2gray
[f,desc2] = sift(I, compute_descriptor=True, float_descriptors=True)
pointsInImage2 = swapcolumn(f[:,0:2])


#%% Matching
M = SIFTSimpleMatcher(desc1, desc2)

#%% Transformation
maxIter = 200
maxInlierErrorPixels = 0.05*len(np.asarray(img1))
seedSetSize = np.max([3, np.ceil(0.1 * len(M))])
minInliersForAcceptance = np.ceil(0.3 * len(M))   
H = RANSACFit(pointsInImage1, pointsInImage2, M, maxIter, seedSetSize, maxInlierErrorPixels, minInliersForAcceptance)


#%% Make Panoramic image
Pano = PairStitch(img1, img2, H, saveFileName)
print('Panorama was saved as uttower_pano.jpg', saveFileName)
plt.imshow(Image.open(saveFileName))
