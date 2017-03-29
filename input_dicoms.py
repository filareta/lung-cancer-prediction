import os
import cv2
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import compress_dicoms as cd
import lung_segmentation as ls


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
IMAGE_PXL_SIZE = 512
# IMAGE_PXL_SIZE = 180
HM_SLICES = 450
# HM_SLICES = 277
THRESHOLD_LOW = -1100.0
THRESHOLD_HIGH = 700.0


def resize(image):
    if not len(image):
        return np.array([])

    return np.stack([cv2.resize(scan, (IMAGE_PXL_SIZE, IMAGE_PXL_SIZE)) for scan in image])


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(l):
    if len(l):
        return sum(l) / len(l)
    return np.full(l.shape, MIN_BOUND, l.dtype)


def get_mean_chunk_slices(slices):
    if len(slices) < HM_SLICES:
        print("New slices are less then required after getting mean images, adding padding.")
        return trim_and_pad(np.array(slices), HM_SLICES) 

    new_slices = []
    for slice_chunk in np.array_split(slices, HM_SLICES):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    # print(len(slices), len(new_slices))
    return np.stack(new_slices)


def read_csv(input_file):
    return pd.read_csv(input_file, index_col=0)


def read_csv_column(input_file, columns=[0]):
    return pd.read_csv(input_file, usecols=columns).values.flatten()


def threshold_and_normalize_scan(scan):
    scan = scan.astype(np.float32)
    scan [scan < THRESHOLD_LOW] = THRESHOLD_LOW
    scan [scan > THRESHOLD_HIGH] = THRESHOLD_HIGH
    
    # Maximum absolute value of any pixel .
    max_abs = abs (max(THRESHOLD_LOW, THRESHOLD_HIGH, key=abs))
    
    # This will bring values between -1 and 1
    scan /= max_abs
    
    return scan


def trim_and_pad(patient_img, slice_count, normalize_pad=True):
    slices, size_x, size_y = patient_img.shape

    if slices == slice_count:
        return patient_img

    if slices > slice_count:
        return patient_img[:slice_count]

    padding = np.full((slice_count-slices, size_x, size_y), MIN_BOUND, patient_img.dtype)
    if normalize_pad:
        padding = normalize(padding)

    return np.vstack([patient_img, padding])


def get_average_shape(dir=cd.COMPRESSED_DICOMS):
    slice_counts = 0
    img_size = 0
    total = 0
    max_slices = 0
    for patient_id in os.listdir(cd.COMPRESSED_DICOMS):
        image = cd.load_patient_image(cd.COMPRESSED_DICOMS, patient_id)
        nodules = ls.get_lung_nodules_candidates(image)
        slices, size, size = nodules.shape
        if slices > max_slices:
            max_slices = slices
        slice_counts += slices
        img_size += size
        total += 1

    # With resampling
    # Max slices:  389
    # Average shape:  (277, 338, 338)

    # Ony segmentation
    # Max slices:  533
    # Average shape:  (170, 512, 512)
    print("Max slices: ", max_slices)
    return (int(slice_counts/total), int(img_size/total), int(img_size/total))


if __name__ == '__main__':
    print ("Average shape: ", get_average_shape())
    for patient_id in os.listdir(cd.COMPRESSED_DICOMS):
        # try:
        # patient_id = '00cba091fa4ad62cc3200a657aeb957e'
        image = cd.load_patient_image(cd.COMPRESSED_DICOMS, patient_id)
    # image = resize(image)

    # image = trim_and_pad(image, HM_SLICES, False)
    # image = normalize(image)
    # image = zero_center(image)
    # image = get_mean_chunk_slices(image)
        print(image.shape)
        scan = image[78]
        print(scan.shape)
        print("Segmented: ", scan)
        plt.imshow(scan, cmap=plt.cm.gray)
        plt.show()
    
        scan[scan < -420] = 0
        print("Thresholded: ", scan)
        plt.imshow(scan, cmap=plt.cm.gray)
        plt.show()
    # cd.store_patient_image('../kaggle-data/segmented-resized', 
        # image, patient_id)
            # print("Patient with id {} has image with shape: {}.".format(patient_id, image.shape))
        # except Exception as e:
            # print("Processing {} failed with {}.".format(patient_id, e))