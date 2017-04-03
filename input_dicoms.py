import os
import cv2
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import compress_dicoms as cd
import lung_segmentation as ls
import config


def resize(image):
    if not len(image):
        return np.array([])

    return np.stack([cv2.resize(scan, (config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y)) 
                     for scan in image])


def normalize(image):
    image = ((image - config.MIN_BOUND) / 
             (config.MAX_BOUND - config.MIN_BOUND))
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(l):
    if len(l):
        return sum(l) / len(l)
    return np.full(l.shape, config.OUT_SCAN, l.dtype)


def get_mean_chunk_slices(slices):
    if len(slices) < config.SLICES:
        print("New slices are less then required after getting mean images, adding padding.")
        return trim_and_pad(np.array(slices), config.SLICES) 

    new_slices = []
    for slice_chunk in np.array_split(slices, config.SLICES):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    return np.stack(new_slices)


def read_csv(input_file):
    return pd.read_csv(input_file, index_col=0)


def read_csv_column(input_file, columns=[0]):
    return pd.read_csv(input_file, usecols=columns).values.flatten()


def trim_and_pad(patient_img, slice_count, normalize_pad=True):
    slices, size_x, size_y = patient_img.shape

    if slices == slice_count:
        return patient_img

    if slices > slice_count:
        return patient_img[:slice_count]

    padding = np.full((slice_count-slices, size_x, size_y), 
        config.OUT_SCAN, patient_img.dtype)
    if normalize_pad:
        padding = normalize(padding)

    return np.vstack([patient_img, padding])


def get_average_shape(directory=config.SEGMENTED_LUNGS_DIR):
    slice_counts = 0
    img_size = 0
    total = 0
    max_slices = 0
    for patient_id in os.listdir(directory):
        image = cd.load_patient_image(directory, patient_id)
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

    # Only segmentation
    # Max slices:  533
    # Average shape:  (170, 512, 512)
    print("Max slices: ", max_slices)
    return (int(slice_counts/total), int(img_size/total), int(img_size/total))
