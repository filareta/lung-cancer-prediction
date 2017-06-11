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

WATERSHED = 2
MORPHOLOGICAL_OPERATIONS = 1


# Input parameters might be able to change those
IMAGE_PXL_SIZE_X = 256
IMAGE_PXL_SIZE_Y = 256
SLICES = 180

SEGMENTED_LUNGS_DIR = 'D:/Fil/segmented_watershed'



IMG_SHAPE = (SLICES, IMAGE_PXL_SIZE_X, 
             IMAGE_PXL_SIZE_Y)


SEGMENTATION_ALGO = WATERSHED