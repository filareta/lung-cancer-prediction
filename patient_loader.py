import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

import utils
import config
import lung_segmentation as ls


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


# Tests with the mean scans loader are not using
# lung segmentation, only compressed sorted slices in HU units.
class MeanScansLoader(PatientImageLoader):
    def __init__(self, images_dir=None):
        super(MeanScansLoader, self).__init__(images_dir)

    def load_scans(self, patient):
        image = utils.load_patient_image(self._images_input, patient)
        image = utils.resize(image)

        return utils.get_mean_chunk_slices(image)

    @property
    def name(self):
        return 'mean_scans_loader'


class SegmentedGaussianLungsLoader(PatientImageLoader):
    def __init__(self, images_dir=config.SEGMENTED_LUNGS_DIR):
        super(SegmentedGaussianLungsLoader, self).__init__(images_dir)

    def process_scans(self, image):
        image = np.stack([cv2.GaussianBlur(scan, (5, 5), 0) for scan in image])
        image = utils.resize(image)

        return utils.trim_pad_slices(image, pad_with_existing=False)

    def load_scans(self, patient):
        image = utils.load_patient_image(self._images_input, patient)
        return self.process_scans(image)

    @property
    def name(self):
        return 'segmented_gaussian_lungs_loader'


# Default loader
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


if __name__ == '__main__':
    loader = SegmentedLungsScansLoader()
    for patient in os.listdir(config.SEGMENTED_LUNGS_DIR):
        lungs = loader.load_scans('026470d51482c93efc18b9803159c960-augm')
        for i, lung in enumerate(lungs):
            print(i)
            plt.imshow(lung, cmap='gray')
            plt.show()



