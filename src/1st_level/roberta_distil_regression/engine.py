import numpy as np
import torch
import tqdm

import utils


def loss_fn(outputs, labels):
    loss_fct = torch.nn.MSELoss()
    return loss_fct(outputs, labels)


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        labels = d['labels']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        model.zero_grad()
        outputs = \
            model(ids=ids, mask=mask)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    rmse_scores = utils.AverageMeter()

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask)
            loss = loss_fn(outputs, labels)

            rmse_scores.update(torch.sqrt(loss), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    print(f'RMSE = {rmse_scores.avg}')
    return rmse_scores.avg
