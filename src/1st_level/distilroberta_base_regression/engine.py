import numpy as np
import torch
import tqdm

import config
import utils


def loss_fn(outputs, labels):
    loss_fct = torch.nn.MSELoss()
    return loss_fct(outputs, labels)


def train_fn(train_data_loader, valid_data_loader, model, optimizer, device, epoch, writer, scheduler=None):
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        labels = d['labels']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        model.train()
        model.zero_grad()
        outputs = \
            model(ids=ids, mask=mask)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=np.sqrt(losses.avg))

        if (bi%config.NUM_SHOW_ITER == 0) and bi!=0 and config.SHOW_ITER_VAL:
            eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer)
    writer.add_scalar('Loss/train', np.sqrt(losses.avg), (epoch+1)*len(train_data_loader))

    rmse_score = engine.eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer)
    return rmse_score

def eval_fn(data_loader, model, device, iteration, writer):
    model.eval()
    losses = utils.AverageMeter()

    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask)
            loss = loss_fn(outputs, labels)

            losses.update(loss.item(), ids.size(0))
    
    writer.add_scalar('Loss/val', np.sqrt(losses.avg), iteration)
    print(f'RMSE iter {iteration}= {np.sqrt(losses.avg)}')
    return np.sqrt(losses.avg)
