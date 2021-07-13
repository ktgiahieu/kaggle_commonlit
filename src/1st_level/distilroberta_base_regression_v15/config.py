import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
if is_kaggle:
    comp_name = 'commonlitreadabilityprize'
    my_impl = 'commonlit-impl'
    my_model_dataset = 'commonlit-distilroberta-base-regression-v15'

    TRAINING_FILE = f'../input/{comp_name}/train.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'../input/{my_model_dataset}'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/distilroberta-base'
else: #colab
    repo_name = 'kaggle_commonlit'
    drive_name = 'Commonlit'
    model_save = 'distilroberta_base_regression_v15'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'distilroberta-base'

# Model params
SEEDS = [25, 42, 123]
N_FOLDS = 5
EPOCHS_P1 = 1
EPOCHS_P2 = 5
LEARNING_RATE_P1 = 5e-5
LEARNING_RATE_P2 = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 256  # actually = inf
MAX_LEN_SENTENCE = 32  # actually = inf
MAX_N_SENTENCE = 24

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

HIDDEN_SIZE = 768
ATTENTION_HIDDEN_SIZE = 512
N_LAST_HIDDEN = 4
N_SENTENCE_FEATURES = 2
N_DOCUMENT_FEATURES = 13
BERT_DROPOUT = 0
CLASSIFIER_DROPOUT = 0
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SHOW_ITER_VAL = False
NUM_SHOW_ITER = 20