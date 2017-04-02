INPUT_DIR = './'
SAMPLE_IMGS = 'D:/Fil/stage1'
COMPRESSED_DICOMS = INPUT_DIR + '/segmented_morph_op'

LABELS_INPUT_DIR = INPUT_DIR + '/input/stage1_labels.csv'
EXACT_TEST_IDS = INPUT_DIR + '/input/stage1_sample_submission.csv'
COLUMN_NAME = 'cancer'
IMG_SHAPE = (SLICES, IMAGE_PXL_SIZE_X, IMAGE_PXL_SIZE_Y, 1)

VALIDATION_IDS = INPUT_DIR + '/input/validation_data.csv'
TRAINING_IDS = INPUT_DIR + '/input/training_data.csv'

# [1, 0]-> no cancer
# [0, 1] -> cancer
CANCER_CLS = 1
NO_CANCER_CLS = 0