import tokenizers
from transformers import AutoTokenizerFast
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
if is_kaggle:
    comp_name = 'commonlitreadabilityprize'
    my_impl = 'my-commonlit-impl'
    my_model_dataset = 'commonlit-roberta-distil-classifier-model'

    TOKENIZER_PATH = f'../input/{my_impl}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'../input/{comp_name}/train.csv'
    TEST_FILE = 'test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'../input/{my_model_dataset}'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/my-distilroberta-base'
else: #colab
    repo_name = 'kaggle_commonlit'
    drive_name = 'Commonlit'
    
    TOKENIZER_PATH = f'/content/{repo_name}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/roberta_distil_classifier'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/roberta_distil_classifier'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    #MODEL_CONFIG = 'huawei-noah/TinyBERT_General_4L_312D'
    MODEL_CONFIG = 'distilroberta-base'

# Model params
SEED = 25
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 5e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 384  # actually = inf

TOKENIZER = AutoTokenizerFast.from_pretrained(
    MODEL_CONFIG)

HIDDEN_SIZE = 768
N_LAST_HIDDEN = 6
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 0.4
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
