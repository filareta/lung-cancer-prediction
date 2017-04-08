import os
import cv2
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def resize(image, 
           x=config.IMAGE_PXL_SIZE_X,
           y=config.IMAGE_PXL_SIZE_Y):
    if not len(image):
        return np.array([])

    return np.stack([cv2.resize(scan, (x, y)) 
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


def store_to_csv(patients, labels, csv_file_path):
    index = pd.Index(data=patients, name=config.ID_COLUMN_NAME)
    df = pd.DataFrame(data={config.COLUMN_NAME: labels}, 
                      columns=[config.COLUMN_NAME],
                      index=index)
    df.to_csv(csv_file_path)


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


def trim_pad_slices(scans, pad_with_existing=True,
                    padding_value=config.BACKGROUND):
    slices, x, y = scans.shape

    if slices == config.SLICES:
        return scans

    if slices < config.SLICES:
        pad = config.SLICES - slices

        if pad_with_existing:
            padding = []
            for slice_chunk in np.array_split(scans, pad):
                # TODO: Think of an improvement, not well sorted 
                # by the slice location
                padding.append(slice_chunk[-1])
        else:
            padding = np.full((pad, x, y), padding_value, scans.dtype)

        return np.vstack([scans, padding])

    trim = slices - config.SLICES
    trimmed = []
    for slice_chunk in np.array_split(scans, trim):
        trimmed.append(slice_chunk[1:])

    return np.vstack(trimmed)


def count_background_rows(image, background=config.BACKGROUND):
    return np.sum(np.all(image == background, axis=1))


def remove_background_rows(image, background=config.BACKGROUND):
    return image[~np.all(image == background, axis=1)]


def remove_background_rows_3d(scans, background=config.BACKGROUND):
    transformed = []
    for scan in scans:
        removed = remove_background_rows(scan, background)
        tr_scan = cv2.resize(removed, 
            (config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y))
        transformed.append(tr_scan)

    return np.stack(transformed)


def store_patient_image(image_dir, image, patient_id):
    """
    Serializes the patient image.

    Image is a 3D numpy array - array from patient slices.
    If not existing image_dir is created.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    np.savez_compressed(os.path.join(image_dir, patient_id), image)


def load_patient_image(image_dir, patient_id):
    """
    Load the serialized patient image.

    Image is a 3D array - array of patient slices, metadata,
    contained in the dicom format, is removed.
    """
    if '.npz' not in patient_id:
        patient_id += '.npz'
    with np.load(os.path.join(image_dir, patient_id)) as data:
        return data['arr_0']