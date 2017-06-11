import config

from data_set import DataLoader
from model import Convolution3DNetwork

import patient_loader as pl
import model_configuration as mc


class ModelFactory(object):
    def __init__(self, selected_model=None):
        self._selected_model = selected_model or config.SELECTED_MODEL
        self._with_augmentation = False
        self._init_model()

    def _init_model(self):
        if self._selected_model == config.BASELINE:
            self._image_loader = pl.MeanScansLoader()
            self._network_config = mc.BaselineConfig()
        elif self._selected_model == config.BASELINE_WITH_SEGMENTATION:
            self._image_loader = pl.SegmentedGaussianLungsLoader()
            self._network_config = mc.BaselineConfig()
        elif self._selected_model == config.NO_REGULARIZATION:
            self._image_loader = pl.SegmentedGaussianLungsLoader()
            self._network_config = mc.NoRegularizationConfig()
        elif self._selected_model == config.NO_REGULARIZATION_WATERSHED:
            # segmentation algorithm has already been selected and
            # changed during config setup
            self._image_loader = pl.SegmentedGaussianLungsLoader()
            self._network_config = mc.NoRegularizationConfig()
        elif self._selected_model == config.DROPOUT_L2NORM_REGULARIZARION:
            self._image_loader = pl.SegmentedGaussianLungsLoader()
            self._network_config = mc.DropoutsWithL2RegularizationConfig()
        elif self._selected_model == config.REGULARIZATION_MORE_SLICES:
            self._image_loader = pl.SegmentedLungsScansLoader()
            self._network_config = mc.DefaultConfig()
        elif self._selected_model == config.WITH_DATA_AUGMENTATION:
            self._image_loader = pl.SegmentedLungsScansLoader()
            self._with_augmentation = True
            self._network_config = mc.DefaultConfig()
        else: #default case
            self._image_loader = pl.SegmentedLungsScansLoader()
            self._with_augmentation = True
            self._network_config = mc.DefaultConfig()

    def get_network_model(self):
        return Convolution3DNetwork(config=self._network_config)

    def get_data_loader(self):
        return DataLoader(images_loader=self._image_loader,
                          add_transformed_positives=self._with_augmentation)