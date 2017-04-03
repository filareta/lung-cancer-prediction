LABELS_INPUT_DIR = './input'
ALL_IMGS = 'D:/Fil/stage1'
SEGMENTED_LUNGS_DIR = '../segmented_morph_op'

PATIENT_LABELS_CSV = LABELS_INPUT_DIR + '/stage1_labels.csv'
TEST_PATIENTS_IDS = LABELS_INPUT_DIR + '/stage1_sample_submission.csv'

VALIDATION_PATINETS_IDS = LABELS_INPUT_DIR + '/validation_data.csv'
TRAINING_PATIENTS_IDS = LABELS_INPUT_DIR + '/training_data.csv'

COLUMN_NAME = 'cancer'
IMG_SHAPE = (SLICES, IMAGE_PXL_SIZE_X, IMAGE_PXL_SIZE_Y, 1)

# [1, 0]-> no cancer
# [0, 1] -> cancer
CANCER_CLS = 1
NO_CANCER_CLS = 0