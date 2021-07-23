import os
import random
import re

import torch
import numpy as np

import config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(model):
    num_layers = 6   #distil
    if config.model_type.split('-')[-1] == 'base':
        num_layers = 12
    elif config.model_type.split('-')[-1]=='large' or config.model_type == 'deberta-v2-xlarge':
        num_layers = 24
    elif  config.model_type == 'deberta-xlarge' or config.model_type == 'deberta-v2-xxlarge':
        num_layers = 48

    named_parameters = list(model.named_parameters()) 
    automodel_parameters = list(model.automodel.named_parameters())
    attention_pooler_parameters = named_parameters[len(automodel_parameters):len(automodel_parameters)+4]
    head_parameters = named_parameters[len(automodel_parameters)+4:]
        
    attention_pooler_group = [params for (name, params) in attention_pooler_parameters]
    head_group = [params for (name, params) in head_parameters]

    parameters = []
    parameters.append({"params": attention_pooler_group,
                       "weight_decay": weight_decay,
                       "lr": config.ATTENTION_POOLER_LEARNING_RATE})
    parameters.append({"params": head_group, "lr": config.HEAD_LEARNING_RATE})

    last_lr = config.LEARNING_RATES_RANGE[0]
    for name, params in automodel_parameters:
        weight_decay = 0.0 if "bias" in name else config.WEIGHT_DECAY
        lr = None

        if config.model_type.split('-')[0] == 'bart':
            found_layer_num_encoder = re.search('(?<=encoder\.layer).*', name)
            if found_layer_num_encoder:
                found_a_number = re.search('(?<=\.)\d+(?=\.)',found_layer_num_encoder.group(0))#fix encoder.layernorm.weight bug
                if found_a_number:
                    layer_num = int(found_a_number.group(0))
                    lr = config.LEARNING_RATES_RANGE[0] + (layer_num+1) * (config.LEARNING_RATES_RANGE[1] - config.LEARNING_RATES_RANGE[0])/num_layers
            
            found_layer_num_decoder = re.search('(?<=decoder\.layer).*', name)
            if found_layer_num_decoder:
                found_a_number = re.search('(?<=\.)\d+(?=\.)',found_layer_num_decoder.group(0))#fix encoder.layernorm.weight bug
                if found_a_number:
                    layer_num = int(found_a_number.group(0))
                    lr = config.LEARNING_RATES_RANGE[0] + (layer_num+13) * (config.LEARNING_RATES_RANGE[1] - config.LEARNING_RATES_RANGE[0])/num_layers

        else:
            found_layer_num = re.search('(?<=encoder\.layer).*', name)
            if found_layer_num:
                layer_num = int(re.search('(?<=\.)\d+(?=\.)',found_layer_num.group(0)).group(0))
                lr = config.LEARNING_RATES_RANGE[0] + (layer_num+1) * (config.LEARNING_RATES_RANGE[1] - config.LEARNING_RATES_RANGE[0])/num_layers

        if lr is None:
            lr = last_lr
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})
        last_lr = lr
    return torch.optim.AdamW(parameters)