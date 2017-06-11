LABELS_INPUT_DIR = './input'
ALL_IMGS = 'D:/Fil/stage1'

PATIENT_LABELS_CSV = LABELS_INPUT_DIR + '/stage1_labels.csv'
TEST_PATIENTS_IDS = LABELS_INPUT_DIR + '/stage1_sample_submission.csv'

VALIDATION_PATINETS_IDS = LABELS_INPUT_DIR + '/validation_data.csv'
TRAINING_PATIENTS_IDS = LABELS_INPUT_DIR + '/training_data.csv'

MODELS_STORE_DIR = './models'
SOLUTION_FILE_PATH = './solution_last.csv'
REAL_SOLUTION_CSV = './input/stage1_solution.csv'
RESTORE_MODEL_CKPT = '/model_best_err14.ckpt'
SUMMARIES_DIR = './summaries/model_summary'
RESTORE = False
START_STEP = 1

COLUMN_NAME = 'cancer'
ID_COLUMN_NAME = 'id'

# [1, 0]-> no cancer
# [0, 1] -> cancer
CANCER_CLS = 1
NO_CANCER_CLS = 0

# Image properties
OUT_SCAN = -2000
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

BACKGROUND = 0
BATCH_SIZE = 1
NUM_CHANNELS = 1
N_CLASSES = 2


# Preprocessing options used for the defined models
BASELINE_PREPROCESS = 0
MORPHOLOGICAL_OPERATIONS = 1
WATERSHED = 2

# TODO: Change paths accordingly after adding download scripts
preprocessed_imgs = {
    BASELINE_PREPROCESS: 'D:/Fil/baseline_preprocessing',
    MORPHOLOGICAL_OPERATIONS: '../kaggle-data/segmented_morph_op',
    WATERSHED: 'D:/Fil/segmented_watershed'
}

# Defined models
BASELINE = 'baseline'
BASELINE_ADDITIONAL_LAYERS = 'baseline_add_layers'
NO_REGULARIZATION = 'no_regularization'
DROPOUT_L2NORM_REGULARIZARION = 'with_regularization'
REGULARIZATION_MORE_SLICES = 'regularization_more_slices'
WITH_DATA_AUGMENTATION = 'more_slices_augmentation'


model_to_img_shape = { 
    BASELINE: (100, 128, 128),
    BASELINE_ADDITIONAL_LAYERS: (100, 128, 128),
    NO_REGULARIZATION: (140, 256, 256),
    DROPOUT_L2NORM_REGULARIZARION: (140, 256, 256),
    REGULARIZATION_MORE_SLICES: (180, 256, 256),
    WITH_DATA_AUGMENTATION: (180, 256, 256)
}

model_to_preprocessing = {
    BASELINE: BASELINE_PREPROCESS,
    BASELINE_ADDITIONAL_LAYERS: BASELINE_PREPROCESS,
    NO_REGULARIZATION: MORPHOLOGICAL_OPERATIONS,
    DROPOUT_L2NORM_REGULARIZARION: WATERSHED,
    REGULARIZATION_MORE_SLICES: WATERSHED,
    WITH_DATA_AUGMENTATION: WATERSHED
}

# This configuration must be changed in order to select other
# predefined model for training
SELECTED_MODEL = WITH_DATA_AUGMENTATION

IMG_SHAPE = model_to_img_shape[SELECTED_MODEL]

SLICES, IMAGE_PXL_SIZE_X, IMAGE_PXL_SIZE_Y = IMG_SHAPE

SEGMENTATION_ALGO = model_to_preprocessing[SELECTED_MODEL]
SEGMENTED_LUNGS_DIR = preprocessed_imgs[SEGMENTATION_ALGO]




