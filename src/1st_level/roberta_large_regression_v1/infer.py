import sys
import pickle
import os
import gc

import numpy as np
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

    test_dataset = dataset.CommonlitDataset(
        texts=df_test.text.values,
        labels=df_test.label.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1)
    
    predicted_labels = []
    for i in range(config.N_FOLDS):  
        print(f'fold{i}')
        all_models = []
        torch.cuda.empty_cache()
        gc.collect()
        for seed in config.SEEDS:
            print(f'adding seed{seed}')
            model = models.CommonlitModel(conf=model_config)
            model.to(device)
            model.load_state_dict(torch.load(
                f'{config.TRAINED_MODEL_PATH}/model_{i}_{seed}.bin'),
                strict=False)
            model.eval()
            all_models.append(model)

        predicted_labels_per_fold = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(tk0):
                ids = d['ids']
                mask = d['mask']
                labels = d['labels']

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)


                outputs_seeds = []
                for s in range(len(config.SEEDS)):
                    print(f'seed{config.SEEDS[s]}')
                    outputs = \
                      all_models[s](ids=ids, mask=mask)

                    outputs_seeds.append(outputs)

                outputs = sum(outputs_seeds) / (len(config.SEEDS))

                outputs = outputs.cpu().detach().numpy()
                predicted_labels_per_fold.extend(outputs.squeeze(-1).tolist())
        predicted_labels.append(predicted_labels_per_fold)
    predicted_labels = np.mean(np.array(predicted_labels), axis=0).tolist()

    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')
        
    pickle_name = sys.argv[1]
    with open(f'{config.INFERED_PICKLE_PATH}/{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please specify output pickle name."
    run()
