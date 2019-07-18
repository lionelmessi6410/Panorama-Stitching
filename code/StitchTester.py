# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:04:00 2017

@author: HGY
"""

import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from cyvlfeat.sift import sift
from SIFTSimpleMatcher import SIFTSimpleMatcher
from RANSACFit import RANSACFit
from MultipleStitch import MultipleStitch
'''
# This script tests your implementation of MultiStitch.m, and you can also
# use it for generating panoramas from your own images.
#
# In case generating a panorama takes too long or too much memory, it is
# advisable to resize images to smaller sizes.
#
# You may also want to tune matching criterion and RANSAC parameters in
# order to get better quality panorama.
'''
def swapcolumn(arr):
    # sweep sift index of [Y,X] -> [X,Y] for easy computation of affine transform
    swap = []
    swap.append(arr[:,1])
    swap.append(arr[:,0])
    swap = np.asarray(swap).T
    return swap


#%% Parameters
Thre = 0.5;
RESIZE = 0.6

#%% Load a list of images (Change file name if you want to use other images)
imgList = glob('../data/Rainier*.png')
saveFileName = '../results/pano.jpg'

#%% Add path
Images = {}
for idx, imgPath in enumerate(sorted(imgList)):
    print(idx, imgPath)
    fileName = os.path.basename(imgPath)
    img = Image.open(imgPath, 'r')
    if (max(img.size)>1000 or len(imgList)>10):
        img.thumbnail((np.asarray(img.size)*RESIZE).astype('int'), Image.ANTIALIAS)
    Images.update({idx: img})
print('Images loaded. Beginning feature detection...')

#%% Feature detection
Descriptor = {}
PointInImg = {}
for idx, (key, img) in enumerate(sorted(Images.items())):
    I = np.asarray(img.convert('L')).astype('single')
    [f,d] = sift(I, compute_descriptor=True, float_descriptors=True)
    pointsInImage = swapcolumn(f[:,0:2])
    PointInImg.update({idx:pointsInImage})
    Descriptor.update({idx:d})

#%% Compute Transformation
Transform = {}
for idx in range(len(imgList)-1):
    print('fitting transformation from '+str(idx)+' to '+str(idx+1)+'\t')
    M = SIFTSimpleMatcher(Descriptor[idx], Descriptor[idx+1], Thre)
    print('matching points:',len(M,),'\n')
    Transform.update({idx:RANSACFit(PointInImg[idx], PointInImg[idx+1], M)})
    
#%% Make Panoramic image
print('Stitching images...')
MultipleStitch(Images, Transform, saveFileName)
print('The completed file has been saved as '+saveFileName)
plt.imshow(Image.open(saveFileName))
