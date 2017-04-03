import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from skimage.morphology import ball, disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np


threshold = -420


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < threshold
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show()

    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plt.imshow(cleared, cmap=plt.cm.bone)
        plt.show()

    '''
    Step 3: Closure operation with a disk of radius 2. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(2)
    binary = binary_closing(cleared, selem)
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show()

    '''    
    Step 4: Label the image.
    '''
    label_image = label(binary)
    if plot == True:
        plt.imshow(label_image, cmap=plt.cm.bone)
        plt.show() 
    
    '''
    Step 5: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]

    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show() 

    '''
    Step 6: Closure operation with a disk of radius 2. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(12)
    binary = binary_closing(binary, selem)
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show()

    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show()

    '''
    Step 8: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plt.imshow(binary, cmap=plt.cm.bone)
        plt.show() 
   
    '''
    Step 9: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plt.imshow(im, cmap=plt.cm.bone)
        plt.show() 
        
    return im

def apply_threshold(threshold, scan):
    scan[scan < threshold] = 0
    return scan

def get_lung_nodules_candidates(patient_imgs):
    nodules = [apply_threshold(threshold, scan) for scan in patient_imgs]
    return np.stack([nodule for nodule in nodules if nodule.any()])