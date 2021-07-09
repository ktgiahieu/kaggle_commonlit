import numpy as np
import torch
import tqdm

import utils


def loss_fn(outputs, labels):
    loss_fct = torch.nn.MSELoss()
    return loss_fct(outputs, labels)


def train_fn(data_loader, valid_data_loader, model, optimizer, device, epoch, writer, scheduler=None):
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
        tk0.set_postfix(loss=np.sqrt(losses.avg))

        if bi%21 == 20:
            eval_iter(valid_data_loader, model, device, epoch*len(data_loader) + bi, writer)
    writer.add_scalar('Loss/train', np.sqrt(losses.avg), (epoch+1)*len(data_loader))

def eval_iter(data_loader, model, device, iteration, writer):
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

def eval_fn(data_loader, model, device, epoch, writer):
    model.eval()
    losses = utils.AverageMeter()

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

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=np.sqrt(losses.avg))
    
    writer.add_scalar('Loss/val', np.sqrt(losses.avg), (epoch+1)*len(data_loader))
    print(f'RMSE = {np.sqrt(losses.avg)}')
    return np.sqrt(losses.avg)
