LABELS_INPUT_DIR = './input'
ALL_IMGS = 'D:/Fil/stage1'

# SEGMENTED_LUNGS_DIR = '../kaggle-data/segmented_morph_op'
SEGMENTED_LUNGS_DIR = '../kaggle-data/segmented_watershed'

PATIENT_LABELS_CSV = LABELS_INPUT_DIR + '/stage1_labels.csv'
TEST_PATIENTS_IDS = LABELS_INPUT_DIR + '/stage1_sample_submission.csv'

VALIDATION_PATINETS_IDS = LABELS_INPUT_DIR + '/validation_data.csv'
TRAINING_PATIENTS_IDS = LABELS_INPUT_DIR + '/training_data.csv'

MODELS_STORE_DIR = './models'
SOLUTION_FILE_PATH = './solution.csv'
REAL_SOLUTION_CSV = './input/stage1_solution.csv'

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
IMAGE_PXL_SIZE_X = 236
IMAGE_PXL_SIZE_Y = 216
SLICES = 160
IMG_SHAPE = (SLICES, IMAGE_PXL_SIZE_X, 
             IMAGE_PXL_SIZE_Y, 1)
BACKGROUND = 0

WATERSHED = 2
MORPHOLOGICAL_OPERATIONS = 1
SEGMENTATION_ALGO = WATERSHED