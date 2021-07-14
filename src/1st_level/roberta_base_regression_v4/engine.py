import numpy as np
import torch
import tqdm

import config
import utils


def loss_fn(outputs, labels):
    loss_fct = torch.nn.MSELoss()
    return loss_fct(outputs, labels)


def train_fn(train_data_loader, valid_data_loader, model, optimizer, device, writer, model_path, scheduler=None):
    losses = utils.AverageMeter()



    best_val_rmse = None
    step = 0
    last_eval_step = 0
    eval_period = config.EVAL_SCHEDULE[0][1]   
    for epoch in range(config.EPOCHS):
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

            if step >= last_eval_step + eval_period:
                val_rmse = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer)                           
                last_eval_step = step
                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break                               
                
                    if not best_val_rmse or val_rmse < best_val_rmse:                    
                        best_val_rmse = val_rmse
                        best_epoch = epoch
                        torch.save(model.state_dict(), model_path)
                        print(f"New best_val_rmse: {best_val_rmse:0.4}")
                    else:       
                        print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                              f"(from epoch {best_epoch})")                                    
                    
            step += 1
            
        writer.add_scalar('Loss/train', np.sqrt(losses.avg), (epoch+1)*len(train_data_loader))

        rmse_score = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer)
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
