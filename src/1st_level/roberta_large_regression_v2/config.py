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
    my_model_dataset = 'commonlit-roberta-large-regression-v2'

    TRAINING_FILE = f'../input/{comp_name}/train.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'../input/{my_model_dataset}'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/roberta-large'
else: #colab
    repo_name = 'kaggle_commonlit'
    drive_name = 'Commonlit'
    model_save = 'roberta_large_regression_v2'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'roberta-large'

EVAL_SCHEDULE = [(0.6, 70),(0.50, 32), (-1., 16)]
# Model params
SEEDS = [1000, 25, 42]
N_FOLDS = 5
EPOCHS = 5

PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
if is_kaggle:
    VALID_BATCH_SIZE = 4
MAX_LEN = 248  # actually = inf

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

HIDDEN_SIZE = 1024
ATTENTION_HIDDEN_SIZE = 1024
N_LAST_HIDDEN = 4
BERT_DROPOUT = 0
CLASSIFIER_DROPOUT = 0
WARMUP_RATIO = 0.125

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SHOW_ITER_VAL = False
NUM_SHOW_ITER = 20

#Author hyperparams
LEARNING_RATES = [2e-5, 3e-5, 4e-5]
WEIGHT_DECAY = 0.01