import sys
import pickle
import os

import pandas as pd
import torch
import transformers
import tqdm

import config
import models
import dataset
import utils


def run():
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, 'label'] = 0
    df_test.rename(columns={'excerpt': 'text'}, inplace=True)

    device = torch.device('cuda')
    model_config = transformers.AutoConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    fold_models = []
    for i in range(config.N_FOLDS):
        model = models.CommonlitModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(
            f'{config.TRAINED_MODEL_PATH}/model_{i}.bin'),
            strict=False)
        model.eval()
        fold_models.append(model)

    test_dataset = dataset.CommonlitDataset(
        texts=df_test.text.values,
        labels=df_test.label.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4)
    
    predicted_labels = []

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)


            outputs_folds = []
            for i in range(config.N_FOLDS):
                outputs = \
                  fold_models[i](ids=ids, mask=mask)

                outputs_folds.append(outputs)

            outputs = sum(outputs_folds) / config.N_FOLDS

            outputs = outputs.cpu().detach().numpy()
            predicted_labels.extend(outputs.squeeze(-1).tolist())

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')
    
    pickle_name = 'roberta-predicted_labels' if (len(sys.argv)<=1) else sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)


if __name__ == '__main__':
    run()
