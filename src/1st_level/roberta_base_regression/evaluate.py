import torch
import numpy as np
import pandas as pd
import transformers
import tqdm.autonotebook as tqdm

import utils
import config
import models
import dataset
import engine


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

	dfx.rename(columns={'excerpt': 'text', 'target': 'label'}, inplace=True)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

	valid_dataset = dataset.CommonlitDataset(
        texts=df_valid.text.values,
        labels=df_valid.label.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True
    model = models.CommonlitModel(conf=model_config)
    model = model.to(device)

    model.load_state_dict(torch.load(
        f'{config.TRAINED_MODEL_PATH}/model_{fold}.bin'))
    model.eval()

    rmse_score = engine.eval_fn(valid_data_loader, model, device)

    return rmse_score


if __name__ == '__main__':
    utils.seed_everything(config.SEED)

    fold_scores = []
    for i in range(config.N_FOLDS):
        fold_score = run(i)
        fold_scores.append(fold_score)

    for i in range(config.N_FOLDS):
        print(f'Fold={i}, RMSE = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')
