import numpy as np
import os

from skimage import measure, morphology, segmentation
from skimage.morphology import ball, disk, binary_erosion, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.segmentation import clear_border

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import scipy.misc

import config


# Optimal threshold, found in an article about segmentation algorithm using
# morphological operations. This is in HU units.
threshold = -420
nodule_threshold = -180
 
#TODO: compare performance of the binary_erosion, binary_closing, binary_dilation
# operations in scipy and skimage libs

class SegmentationAlgorithm(object):
    def __init__(self, threshold):
        self._threshold = threshold

    def get_segmented_lungs(self, plot=False):
        pass

    def apply_threshold(self, scan):
        scan[scan < nodule_threshold] = 0
        return scan

    def get_lung_nodules_candidates(self, patient_imgs):
        nodules = [self.apply_threshold(scan) for scan in patient_imgs]
        return np.stack([nodule for nodule in nodules if nodule.any()])

    def get_slices_with_nodules(self, patient_imgs):
        return np.stack([slice_ for slice_ in patient_imgs if 
            self._has_nodule(slice_)])

    def _has_nodule(self, scan):
        scan_copy = scan.copy()
        scan_copy[scan_copy < nodule_threshold] = 0
        return scan_copy.any()


class MorphologicalSegmentation(SegmentationAlgorithm):
    def __init__(self, threshold=-420):
        super(MorphologicalSegmentation, self).__init__(threshold)

    def get_segmented_lungs(self, im, plot=False):
        '''
        This funtion segments the lungs from the given 2D slice.
        '''
        '''
        Step 1: Convert into a binary image. 
        '''
        binary = im < self._threshold
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


class WatershedSegmentation(SegmentationAlgorithm):
    def __init__(self, threshold=-400):
        super(WatershedSegmentation, self).__init__(threshold)

    def get_segmented_lungs(self, image, plot=False):
        # TODO: might add the logic for plotting the filters applied in the process
        #Creation of the markers as shown above:
        marker_internal, marker_external, marker_watershed = self.generate_markers(image)
        
        #Creation of the Sobel-Gradient
        sobel_filtered_dx = ndi.sobel(image, 1)
        sobel_filtered_dy = ndi.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        sobel_gradient *= 255.0 / np.max(sobel_gradient)
        
        #Watershed algorithm
        watershed = morphology.watershed(sobel_gradient, marker_watershed)
        
        #Reducing the image created by the Watershed algorithm to its outline
        outline = ndi.morphological_gradient(watershed, size=(3,3))
        outline = outline.astype(bool)
        
        #Performing Black-Tophat Morphology for reinclusion
        #Creation of the disk-kernel and increasing its size a bit
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]
        blackhat_struct = ndi.iterate_structure(blackhat_struct, 8)
        #Perform the Black-Hat
        outline += ndi.black_tophat(outline, structure=blackhat_struct)
        
        #Use the internal marker and the Outline that was just created to generate the lungfilter
        lungfilter = np.bitwise_or(marker_internal, outline)
        #Close holes in the lungfilter
        #fill_holes is not used here, since in some slices the heart would be reincluded by accident
        lungfilter = ndi.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
        
        #Apply the lungfilter (the filtered areas being assigned 0 HU
        segmented = np.where(lungfilter == 1, image, np.zeros(image.shape))

        return segmented


    def generate_markers(self, image):
        #Creation of the internal Marker
        marker_internal = image < self._threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                           marker_internal_labels[coordinates[0], coordinates[1]] = 0
        marker_internal = marker_internal_labels > 0
        #Creation of the external Marker
        external_a = ndi.binary_dilation(marker_internal, iterations=10)
        external_b = ndi.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        #Creation of the Watershed Marker matrix
        marker_watershed = np.zeros(image.shape, dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        
        return marker_internal, marker_external, marker_watershed


def get_segmentation_algorithm():
    if config.SEGMENTATION_ALGO == config.MORPHOLOGICAL_OPERATIONS:
        return MorphologicalSegmentation()
    if config.SEGMENTATION_ALGO == config.WATERSHED:
        return WatershedSegmentation()

    # default for now
    return MorphologicalSegmentation()