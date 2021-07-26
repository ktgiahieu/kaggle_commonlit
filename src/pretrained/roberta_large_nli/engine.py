import numpy as np
import torch
import tqdm

import config
import utils

def loss_fn(outputs, labels):
    loss_fct = torch.nn.BCELoss()
    return loss_fct(outputs, labels)


def train_fn(train_data_loader, valid_data_loader, model, optimizer, device, writer, model_path, scheduler=None):
    best_val_rmse = None
    step = 0
    last_eval_step = 0
    eval_period = config.EVAL_SCHEDULE[0][1]   
    for epoch in range(config.EPOCHS):
        losses = utils.AverageMeter()
        tk0 = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        for bi, d in enumerate(tk0):
            ids_x = d['ids_x']
            ids_y = d['ids_y']
            mask_x = d['mask_x']
            mask_y = d['mask_y']
            labels = d['labels']

            ids_x = ids_x.to(device, dtype=torch.long)
            ids_y = ids_y.to(device, dtype=torch.long)
            mask_x = mask_x.to(device, dtype=torch.long)
            mask_y = mask_y.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            model.train()
            model.zero_grad()
            outputs = model(ids_x=ids_x, ids_y=ids_y, mask_x=mask_x, mask_y=mask_y)
        
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.update(loss.item(), ids_x.size(0))
            tk0.set_postfix(loss=np.sqrt(losses.avg))

            #if step >= last_eval_step + eval_period:
            #    val_rmse = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) + bi, writer)                           
            #    last_eval_step = step
            #    for rmse, period in config.EVAL_SCHEDULE:
            #        if val_rmse >= rmse:
            #            eval_period = period
            #            break                               
                
            #    if not best_val_rmse or val_rmse < best_val_rmse:                    
            #        best_val_rmse = val_rmse
            #        best_epoch = epoch
            #        torch.save(model.state_dict(), model_path)
            #        print(f"New best_val_rmse: {best_val_rmse:0.4}")
            #    else:       
            #        print(f"Still best_val_rmse: {best_val_rmse:0.4}",
            #                f"(from epoch {best_epoch})")                                    
                    
            #step += 1
            if bi%101 == 100:
                val_rmse = eval_fn(valid_data_loader, model, device, epoch*len(train_data_loader) +bi, writer)
                if not best_val_rmse or val_rmse < best_val_rmse:                    
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    model.save_pretrained(model_path)
                    print(f"New best_val_rmse: {best_val_rmse:0.4}")
                else:       
                    print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                            f"(from epoch {best_epoch})")
                writer.add_scalar('Loss/train', losses.avg, (epoch+1)*len(train_data_loader))
    
        writer.add_scalar('Loss/train', losses.avg, (epoch+1)*len(train_data_loader))

        val_rmse = eval_fn(valid_data_loader, model, device, (epoch+1)*len(train_data_loader), writer)
        if not best_val_rmse or val_rmse < best_val_rmse:                    
            best_val_rmse = val_rmse
            best_epoch = epoch
            model.save_pretrained(model_path)
            print(f"New best_val_rmse: {best_val_rmse:0.4}")
        else:       
            print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                    f"(from epoch {best_epoch})")
        writer.add_scalar('Loss/train', losses.avg, (epoch+1)*len(train_data_loader))
        valid_loss = val_rmse

    config.TOKENIZER.save_pretrained(model_path)
    return valid_loss

def eval_fn(data_loader, model, device, iteration, writer):
    model.eval()
    losses = utils.AverageMeter()

    acc = 0
    len_sample = 0
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids_x = d['ids_x']
            ids_y = d['ids_y']
            mask_x = d['mask_x']
            mask_y = d['mask_y']
            labels = d['labels']

            ids_x = ids_x.to(device, dtype=torch.long)
            ids_y = ids_y.to(device, dtype=torch.long)
            mask_x = mask_x.to(device, dtype=torch.long)
            mask_y = mask_y.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            model.train()
            model.zero_grad()
            outputs = model(ids_x=ids_x, ids_y=ids_y, mask_x=mask_x, mask_y=mask_y)

            loss = loss_fn(outputs, labels)

            losses.update(loss.item(), ids_x.size(0))

            acc += torch.sum(torch.abs(outputs - labels) < 0.5).item()
            len_sample += ids_x.size(0)
    
    acc /= len_sample
    writer.add_scalar('Acc/val', acc, iteration)
    print(f'Valid acc iter {iteration}= {acc}')
    writer.add_scalar('Loss/val', losses.avg, iteration)
    print(f'Valid loss iter {iteration}= {losses.avg}')
    return losses.avg
