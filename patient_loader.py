import input_dicoms as ind
import compress_dicoms as cd
import lung_segmentation as ls
import numpy as np
import os


class PatientImageLoader(object):
    def __init__(self, images_dir):
        self._images_input = images_dir

    def load_scans(self, patient):
        pass

    @property
    def images_input(self):
        return self._images_input


class MeanScansLoader(PatientImageLoader):
    def __init__(self, images_dir=cd.COMPRESSED_DICOMS):
        super(MeanScansLoader, self).__init__(images_dir)

    def load_scans(self, patient):
        image = cd.load_patient_image(self._images_input, patient)
        if len(image) == 0:
            print("Empty image! " + patient)
        # image = ind.resize(image)
        # image = ind.threshold_and_normalize_scan(image)
        return ind.get_mean_chunk_slices(image)
        

class TrimPadScansLoader(PatientImageLoader):
    def __init__(self, images_dir=cd.COMPRESSED_DICOMS):
        super(TrimPadScansLoader, self).__init__(images_dir)

    def load_scans(self, patient):
        image = cd.load_patient_image(self._images_input, patient)
        image = ind.trim_and_pad(image, ind.HM_SLICES, False)
        image = ind.threshold_and_normalize_scan(image)
        
        return ind.resize(image)


class NodulesScansLoader(PatientImageLoader):
    def __init__(self, images_dir=cd.COMPRESSED_DICOMS):
        super(NodulesScansLoader, self).__init__(images_dir)

    def process_scans(self, patient):
        image = cd.load_patient_image(self._images_input, patient)
        nodules = ls.get_lung_nodules_candidates(image)
        
        # TODO: Could be improved
        slices, x, y = nodules.shape

        if slices == ind.HM_SLICES:
            return nodules

        if slices < ind.HM_SLICES:
            pad = ind.HM_SLICES - slices
            # print("Nodules less than required slices, add padding with ", pad)
            padding = []
            for slice_chunk in np.array_split(nodules, pad):
                # TODO: Think of an improvement, not well sorted by the slice location
                padding.append(slice_chunk[-1])

            return np.vstack([nodules, padding])

        trim = slices - ind.HM_SLICES
        # print("Nodules more than required, trim with ", trim)
        trimmed = []
        for slice_chunk in np.array_split(nodules, trim):
            trimmed.append(slice_chunk[1:])

        return np.vstack(trimmed)

    def load_scans(self, patient):
        return self.process_scans(patient)


if __name__ == '__main__':
    loader = NodulesScansLoader()
    for patient in os.listdir(cd.COMPRESSED_DICOMS):
    # patient = '008464bb8521d09a42985dd8add3d0d2'
        lungs = loader.process_scans(patient)


