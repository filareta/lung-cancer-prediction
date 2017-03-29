import multiprocessing
import concurrent.futures
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import dicom
import pylab
import scipy.ndimage as spi
import lung_segmentation as ls


INPUT_DIR = './'
SAMPLE_IMGS = '../kaggle-data/stage1'
COMPRESSED_DICOMS = INPUT_DIR + '/segmented_morph_op'

NUM_PROCESSES = multiprocessing.cpu_count()


def load_scans(patient):
    patient_dir = os.path.join(SAMPLE_IMGS, patient)
    slices = [dicom.read_file(os.path.join(patient_dir, scan)) for scan in os.listdir(patient_dir)]

    # ImagePositionPatient[2] equals the slice location == Z coordinate of the scan
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# Resampling to an isomorphic resolution in order to remove variance in the scans
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = spi.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def store_patient_image(image_dir, image, patient_id):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    np.savez_compressed(os.path.join(image_dir, patient_id), image)


def load_patient_image(image_dir, patient_id):
    if '.npz' not in patient_id:
        patient_id += '.npz'
    with np.load(os.path.join(image_dir, patient_id)) as data:
        return data['arr_0']


def process_patients_chunk(patients):
    for patient in patients:
        try:
            scans = load_scans(patient)
            patient_imgs = get_pixels_hu(scans)

            segmented_lungs = np.stack([ls.get_segmented_lungs(image)
                                        for image in patient_imgs])
            # Excluse resampling for now, could be done after if needed.
            # resizing and normalization could be pretraining transforms
            store_patient_image(COMPRESSED_DICOMS, segmented_lungs, patient)
            print("====== Store patient {} image ======.".format(patient))
        except Exception as e:
            print("An error occured while processing {}! {}".format(patient, e))



def process_dicom_set(input_dir):
    patients = np.array([patient for patient in os.listdir(input_dir)])
    chunked_data = np.array_split(patients, NUM_PROCESSES)
    print("Number of processes: {}, total chunks of data {}!".format(NUM_PROCESSES, len(chunked_data)))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES)

    futures = []
    for i, data in enumerate(chunked_data):
        try:
            f = executor.submit(process_patients_chunk, data)
            print("Submit {} batch to executor!". format(i))
            futures.append(f)
        except Exception as e:
            print("An error occured while processing data chunk with size {} on iteration {}: {}".format(len(data), str(i), e))

    print(concurrent.futures.wait(futures)) # By defaults waits for all
    print("Shutdown and wait for processes!")
    executor.shutdown(wait=True)


if __name__ == '__main__':
    process_dicom_set(SAMPLE_IMGS)