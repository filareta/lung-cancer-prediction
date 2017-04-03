import os
import numpy as np

import utils
import config
import lung_segmentation as ls


class PatientImageLoader(object):
    def __init__(self, images_dir):
        self._images_input = images_dir or config.SEGMENTED_LUNGS_DIR

    def load_scans(self, patient):
        pass

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
        nodules = ls.get_lung_nodules_candidates(image)
        
        # TODO: Could be improved
        slices, x, y = nodules.shape

        if slices == config.SLICES:
            return nodules

        if slices < config.SLICES:
            pad = config.SLICES - slices

            padding = []
            for slice_chunk in np.array_split(nodules, pad):
                # TODO: Think of an improvement, not well sorted 
                # by the slice location
                padding.append(slice_chunk[-1])

            return np.vstack([nodules, padding])

        trim = slices - config.SLICES
        trimmed = []
        for slice_chunk in np.array_split(nodules, trim):
            trimmed.append(slice_chunk[1:])

        return np.vstack(trimmed)

    def load_scans(self, patient):
        return self.process_scans(patient)

    @property
    def name(self):
        return 'nodules_scans_loader'


if __name__ == '__main__':
    loader = NodulesScansLoader()
    for patient in os.listdir(config.SEGMENTED_LUNGS_DIR):
    # patient = '008464bb8521d09a42985dd8add3d0d2'
        lungs = loader.process_scans(patient)


