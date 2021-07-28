import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
model_type = 'deberta-xlarge'
comp_name = 'commonlitreadabilityprize'
my_impl = 'commonlit-impl'
my_model_dataset = 'commonlit-deberta-xlarge-meanmax'
if is_kaggle:
    TRAINING_FILE = f'../input/{comp_name}/train.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'../input/{my_model_dataset}'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/deberta-xlarge'
else: #colab
    repo_name = 'kaggle_commonlit'
    drive_name = 'Commonlit'
    model_save = 'deberta_xlarge_meanmax'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'microsoft/deberta-xlarge'


# Model params
SEEDS = [1000, 25, 42]
N_FOLDS = 5
EPOCHS = 4

PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
ACCUMULATION_STEPS = 1
MAX_LEN = 192  # actually = inf

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

EVAL_SCHEDULE = [
                (0.6, 280*ACCUMULATION_STEPS),
                (0.50, 128*ACCUMULATION_STEPS), 
                (0.49, 64*ACCUMULATION_STEPS), 
                (0.48, 48*ACCUMULATION_STEPS), 
                (0.47, 32*ACCUMULATION_STEPS), 
                (-1., 24*ACCUMULATION_STEPS)
                ]

HIDDEN_SIZE = 1024
ATTENTION_HIDDEN_SIZE = 1024
N_LAST_HIDDEN = 4
BERT_DROPOUT = 0
CLASSIFIER_DROPOUT = 0
WARMUP_RATIO = 0.25

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SHOW_ITER_VAL = False
NUM_SHOW_ITER = 20

#Author hyperparams
HEAD_LEARNING_RATE = 1e-3
LEARNING_RATES_RANGE = [3e-6, 8e-6]
WEIGHT_DECAY = 0.01