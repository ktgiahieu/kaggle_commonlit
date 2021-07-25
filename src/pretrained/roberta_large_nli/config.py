import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import tokenizers
from transformers import AutoTokenizer
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
model_type = 'roberta-large'
comp_name = 'commonlitreadabilityprize'
my_impl = 'commonlit-impl'
my_model_dataset = 'commonlit-roberta-large-nli'
if is_kaggle:
    TRAINING_FILE = f'../input/{my_impl}/data/train_nli.csv'
    VALID_FILE = f'../input/{my_impl}/data/valid_nli.csv'
    TEST_FILE = f'../input/{comp_name}/test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'.'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/roberta-large'
else: #colab
    repo_name = 'kaggle_commonlit'
    drive_name = 'Commonlit'
    model_save = 'roberta-large-nli'
    
    TRAINING_FILE = f'/content/{repo_name}/data/train_nli.csv'
    VALID_FILE = f'/content/{repo_name}/data/valid_nli.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/{model_save}'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/pretrained/{model_save}'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'roberta-large'

# Model params
SEEDS = [1000]
EPOCHS = 1

PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
MAX_LEN = 248

EVAL_SCHEDULE = [
                (0.6, 70*ACCUMULATION_STEPS),
                (0.50, 16*ACCUMULATION_STEPS), 
                (0.49, 8*ACCUMULATION_STEPS), 
                (0.48, 4*ACCUMULATION_STEPS), 
                (0.47, 2*ACCUMULATION_STEPS), 
                (-1., 1*ACCUMULATION_STEPS)
                ]

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_CONFIG)

HIDDEN_SIZE = 1024
ATTENTION_HIDDEN_SIZE = 1024
N_LAST_HIDDEN = 4
BERT_DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.1
WARMUP_RATIO = 0.125

USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

SHOW_ITER_VAL = False
NUM_SHOW_ITER = 20

#Author hyperparams
HEAD_LEARNING_RATE = 1e-3
LEARNING_RATES_RANGE = [2e-5, 4e-5]
WEIGHT_DECAY = 0.01