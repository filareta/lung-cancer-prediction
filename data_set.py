import os
import numpy as np
import random as rnd

import patient_loader as pl
import config
from utils import read_csv_column, read_csv


class DataLoader(object):
    def __init__(self, 
                 images_loader=None, 
                 labels_input=config.PATIENT_LABELS_CSV,
                 exact_tests=config.TEST_PATIENTS_IDS,
                 train_set=config.TRAINING_PATIENTS_IDS,
                 validation_set=config.VALIDATION_PATINETS_IDS):
        self._images_loader = images_loader or pl.SegmentedLungsScansLoader()
        self._labels = read_csv(labels_input)

        self._exact_tests = []
        if exact_tests:
            self._exact_tests = read_csv_column(exact_tests)

        self._train_set = list(read_csv_column(train_set, 
                                               columns=[1]))
        self._validation_set = list(read_csv_column(
            validation_set, columns=[1]))

        self._examples_count = len(self._validation_set) + len(self._train_set)
        print("Total examples used for training and validation: ",
            self._examples_count)
        
        print("Total patients used for validation: ", 
            len(self._validation_set))
        print("Total patients used for training: ", 
            len(self._train_set))
        self._exact_tests_count = len(self._exact_tests)

    def _double_positive_class_data(self):
        positive = self.patients_from_class(self._train_set, 
                                            config.CANCER_CLS)
        print("Patients with cancer are: {}".format(len(positive)))
        self._train_set.extend(positive)

    def patients_from_class(self, patient_ids, clazz):
        return [patient for patient in patient_ids 
                if self.get_label(patient) == clazz]

    @property
    def exact_tests_count(self):
        return self._exact_tests_count

    @property
    def examples_count(self):
        return self._examples_count
    
    def train_samples_count(self):
        return len(self._train_set)
    
    def validation_samples_count(self):
        return len(self._validation_set)

    def get_label(self, patient_id):
        try:
            #[first class=no cancer=0, second class=cancer=1]
            # [1, 0]-> no cancer
            # [0, 1] -> cancer
            clazz = self._labels.get_value(patient_id, config.COLUMN_NAME)
            return clazz
        except KeyError as e:
            print("No key found for patient with id {} in the labels.".format(
                   patient_id))
        return None

    def has_label(self, patient):
        try:
            self._labels.get_value(patient, config.COLUMN_NAME)
        except KeyError as e:
            return False
        return True

    def get_training_set(self):
        return DataSet(self._train_set, self)

    def get_validation_set(self):
        return DataSet(self._validation_set, self)

    def get_test_set(self):
        return DataSet(self._test_set, self)

    def get_exact_tests_set(self):
        return DataSet(self._exact_tests, self)

    def load_image(self, patient):
        scans = self._images_loader.load_scans(patient)
        if len(scans):
            return scans.reshape(*config.IMG_SHAPE)
        return scans

    def results_out_dir(self):
        out_dir = os.path.join(config.MODELS_STORE_DIR,
            self._images_loader.name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        return out_dir


class DataSet(object):
    def __init__(self, data_set, data_loader):
        self._data_set = data_set
        self._data_loader = data_loader
        self._index_in_epoch = 0
        self._finished_epochs = 0
        self._num_samples = len(self._data_set)

    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch

        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self._num_samples:
            # Epoche has finished, start new iteration
            self._finished_epochs += 1
            start = 0
            self._index_in_epoch = batch_size
            # Shuffle data
            rnd.shuffle(self._data_set)
        
        end = self._index_in_epoch

        data_set, labels = [], []

        while start < end and end < self._num_samples:
            patient = self._data_set[start]
            image, label = self._patient_with_label(patient)
          
            if len(image) and label is not None:
                labels.append(label)
                data_set.append(image)
            else:
                end += 1
            start +=1

        self._index_in_epoch = end
        if len(data_set) < batch_size:
            print("Current batch size is less: {}".format(len(data_set)))
            print("Start {}, end {}, samples {}".format(start, end, 
                self._num_samples))
        return data_set, labels

    # Used during exact testing phase, here no labels are returned
    def next_patient(self):
        if self._index_in_epoch >= self._num_samples:
            return (None, [])

        patient_id = self._data_set[self._index_in_epoch]
        self._index_in_epoch += 1
        image = self._load_patient(patient_id)
        if self._validate_input_shape(image):
            return (patient_id, image)
        return (patient_id, [])

    def _patient_with_label(self, patient_id):
        label = self._data_loader.get_label(patient_id)
        if label is None:
            return ([], None)
        
        image = self._load_patient(patient_id)
        if self._validate_input_shape(image):
            return (image, label)
        
        return ([], None)

    def _load_patient(self, patient):
        return self._data_loader.load_image(patient)

    def _validate_input_shape(self, patient_image):
        return patient_image.shape == config.IMG_SHAPE

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def finished_epochs(self):
        return self._finished_epochs
    

if __name__ == '__main__':
    data_loader = DataLoader()