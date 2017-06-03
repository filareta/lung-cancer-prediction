import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

import utils
import config
import lung_segmentation as ls
import preprocess_dicoms as pd


segmentation_algo = ls.get_segmentation_algorithm()


class PatientImageLoader(object):
    def __init__(self, images_dir):
        self._images_input = images_dir or config.SEGMENTED_LUNGS_DIR
        self._augment = False

    def load_scans(self, patient):
        if 'augm' in patient:
            self._augment = True
            patient = patient.split('-')[0]
        return utils.load_patient_image(self._images_input, patient)

    @property
    def images_input(self):
        return self._images_input

    @property
    def name(self):
        return 'base_image_loader'


# Tests with the mean scans loader were done without
# lung segmentation, only compressed sorted slices in HU units.
class MeanScansLoader(PatientImageLoader):
    def __init__(self, images_dir):
        super(MeanScansLoader, self).__init__(images_dir)

    def load_scans(self, patient):
        image = utils.load_patient_image(self._images_input, patient)
        return utils.get_mean_chunk_slices(image)

    @property
    def name(self):
        return 'mean_scans_loader'
        

class TrimPadScansLoader(PatientImageLoader):
    def __init__(self, images_dir=config.SEGMENTED_LUNGS_DIR):
        super(TrimPadScansLoader, self).__init__(images_dir)

    def load_scans(self, patient):
        image = utils.load_patient_image(self._images_input, patient)
        image = utils.trim_and_pad(image, config.SLICES, False)
        
        return utils.resize(image)

    @property
    def name(self):
        return 'trim_pad_scans_loader'


class NodulesScansLoader(PatientImageLoader):
    def __init__(self, images_dir=config.SEGMENTED_LUNGS_DIR):
        super(NodulesScansLoader, self).__init__(images_dir)

    def process_scans(self, patient):
        image = utils.load_patient_image(self._images_input, patient)
        nodules = segmentation_algo.get_lung_nodules_candidates(image)
        nodules = utils.resize(nodules)

        return utils.trim_pad_slices(nodules)

    def load_scans(self, patient):
        return self.process_scans(patient)

    @property
    def name(self):
        return 'nodules_scans_loader'


class SegmentedLungsScansLoader(PatientImageLoader):
    def __init__(self, images_dir=config.SEGMENTED_LUNGS_DIR):
        super(SegmentedLungsScansLoader, self).__init__(images_dir)

    def process_scans(self, image):
        image = segmentation_algo.get_slices_with_nodules(image)
        image = utils.resize(image)

        if self._augment:
            angle = randrange(-15, 15)
            image = utils.rotate_scans(image, angle)

        return utils.trim_pad_slices(image, pad_with_existing=True)

    def load_scans(self, patient):
        image = super(SegmentedLungsScansLoader, self).load_scans(patient)
        return self.process_scans(image)

    @property
    def name(self):
        return 'segmented_lungs_loader_with_augmentation'


class CroppedLungScansLoader(PatientImageLoader):
    def __init__(self, images_dir=config.SEGMENTED_LUNGS_DIR):
        super(CroppedLungScansLoader, self).__init__(images_dir)

    def process_scans(self, patient):
        image = np.array([])
        try:
            image = utils.load_patient_image(self._images_input, patient)
        except Exception as e:
            print("Could not load image {}".format(e))
            return image

        nodules = segmentation_algo.get_lung_nodules_candidates(image)
        # Involves resizing currently, removing background must be improved
        nodules = utils.remove_background_rows_3d(nodules)

        return utils.trim_pad_slices(nodules, pad_with_existing=False)
    
    def load_scans(self, patient):
        return self.process_scans(patient)

    @property
    def name(self):
        return 'cropped_nodules_scans_loader'


if __name__ == '__main__':
    # loader = CroppedLungScansLoader()
    loader = SegmentedLungsScansLoader()
    for patient in os.listdir(config.SEGMENTED_LUNGS_DIR):
        lungs = loader.load_scans('026470d51482c93efc18b9803159c960-augm')
        original = pd.get_pixels_hu(pd.load_scans('026470d51482c93efc18b9803159c960'))
        for i, lung in enumerate(lungs):
            print(i)
            plt.imshow(lung, cmap='gray')
            plt.show()
            plt.imshow(original[i], cmap='gray')
            plt.show()



