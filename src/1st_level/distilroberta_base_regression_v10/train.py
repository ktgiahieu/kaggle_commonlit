import os
import numpy as np
import pandas as pd
import transformers
import torch
import torchcontrib
from torch.utils.tensorboard import SummaryWriter
writer = None

import config
import dataset
import models
import engine
import utils


def run(fold, seed):
    dfx = pd.read_csv(config.TRAINING_FILE)
    
    dfx.rename(columns={'excerpt': 'text', 'target': 'label'}, inplace=True)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.CommonlitDataset(
        texts=df_train.text.values,
        labels=df_train.label.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True)

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
    ##
    model_config.hidden_dropout_prob = config.BERT_DROPOUT
    ##
    model = models.CommonlitModel(conf=model_config)
    model = model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    num_train_steps_p1 = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS_P1)
    base_opt_p1 = transformers.AdamW(optimizer_parameters,
                                  lr=config.LEARNING_RATE_P1)
    optimizer_p1 = torchcontrib.optim.SWA(
        base_opt_p1,
        swa_start=int(num_train_steps_p1 * config.SWA_RATIO),
        swa_freq=config.SWA_FREQ,
        swa_lr=None)
    scheduler_p1 = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer_p1,
        num_warmup_steps=int(num_train_steps_p1 * config.WARMUP_RATIO),
        num_training_steps=num_train_steps_p1)

    num_train_steps_p2 = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS_P2)
    base_opt_p2 = transformers.AdamW(optimizer_parameters,
                                  lr=config.LEARNING_RATE_P2)
    optimizer_p2 = torchcontrib.optim.SWA(
        base_opt_p2,
        swa_start=int(num_train_steps_p2 * config.SWA_RATIO),
        swa_freq=config.SWA_FREQ,
        swa_lr=None)
    scheduler_p2 = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer_p2,
        num_warmup_steps=int(num_train_steps_p2 * config.WARMUP_RATIO),
        num_training_steps=num_train_steps_p2)

    print(f'Training is starting for fold={fold}')
    print(f'Training phase 1')
    for epoch in range(config.EPOCHS_P1):
        for param in model.automodel.parameters():
            param.requires_grad = False
        rmse_score = engine.train_fn(train_data_loader, valid_data_loader, model, optimizer_p1,
                        device, epoch, writer, scheduler=scheduler)
    print(f'Training phase 2')
    for epoch in range(config.EPOCHS_P2):
        for param in model.automodel.parameters():
            param.requires_grad = True
        rmse_score = engine.train_fn(train_data_loader, valid_data_loader, model, optimizer_p2,
                        device, epoch, writer, scheduler=scheduler)

    if config.USE_SWA:
        optimizer.swap_swa_sgd()

    if not os.path.isdir(f'{config.MODEL_SAVE_PATH}'):
        os.makedirs(f'{config.MODEL_SAVE_PATH}')

    torch.save(model.state_dict(),
               f'{config.MODEL_SAVE_PATH}/model_{fold}_{seed}.bin')

    return rmse_score


if __name__ == '__main__':
    for seed in config.SEEDS:
        utils.seed_everything(seed=seed)
        print(f"Training with SEED={seed}")
        fold_scores = []
        for i in range(config.N_FOLDS):
            writer = SummaryWriter(f"logs/fold{i}_seed{seed}")
            fold_score = run(i, seed)
            fold_scores.append(fold_score)
            writer.close()

        print('\nScores without SWA:')
        for i in range(config.N_FOLDS):
            print(f'Fold={i}, RMSE = {fold_scores[i]}')
        print(f'Mean = {np.mean(fold_scores)}')
        print(f'Std = {np.std(fold_scores)}')
