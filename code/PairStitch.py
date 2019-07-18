# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:05:26 2017

@author: HGY
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#%% PairStitch function
def PairStitch(img1, img2, H, fileName='pano.jpg'):
    '''
    PairStitch Stitch a pair image.
    Stitch img1 to img2 given the transformation from img1 to img2 is H.
    Save the stitched panorama to fileName.
        
    INPUT:
    - img1: image 1
    - img2: image 2
    - H: 3 by 3 affine transformation matrix
    - fileName: specified file name
    
    OUTPUT:
    - Pano: the panoramic image
    '''
    
    nrows, ncols, _ = np.asarray(img1).shape
    Hinv = np.linalg.inv(H)
    Hinvtuple = (Hinv[0,0],Hinv[0,1], Hinv[0,2], Hinv[1,0],Hinv[1,1],Hinv[1,2])
    Pano = np.asarray(img1.transform((ncols*3,nrows*3), Image.AFFINE, Hinvtuple))
    Pano.setflags(write=1)
    plt.imshow(Pano)
    
    Hinv = np.linalg.inv(np.eye(3))
    Hinvtuple = (Hinv[0,0],Hinv[0,1], Hinv[0,2], Hinv[1,0],Hinv[1,1],Hinv[1,2])
    AddOn = np.asarray(img2.transform((ncols*3,nrows*3), Image.AFFINE, Hinvtuple))
    AddOn.setflags(write=1)
    plt.imshow(AddOn)

    result_mask = np.sum(Pano, axis=2) != 0
    temp_mask = np.sum(AddOn, axis=2) != 0
    add_mask = temp_mask | ~result_mask
    for c in range(Pano.shape[2]):
        cur_im = Pano[:,:,c]
        temp_im = AddOn[:,:,c]
        cur_im[add_mask] = temp_im[add_mask]
        Pano[:,:,c] = cur_im
    plt.imshow(Pano)
    
    
    # Cropping
    boundMask = np.where(np.sum(Pano, axis=2) != 0)
    Pano = Pano[:np.max(boundMask[0]),:np.max(boundMask[1])]
    # plt.imshow(Pano)
    
    # Savefig
    result = Image.fromarray(Pano)
    result.save(fileName)
    
    return Pano
