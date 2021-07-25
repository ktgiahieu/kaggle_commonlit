import numpy as np
import torch

import config

def process_data(text_x, text_y, label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    encoded_dict_x = tokenizer.encode_plus(
        text_x,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        padding = 'max_length',
		return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = True,
    )
    encoded_dict_y = tokenizer.encode_plus(
        text_y,
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        padding = 'max_length',
		return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = True,
    )
    # ----------------------------------

    # Input for BERT
    input_ids_x = np.squeeze(encoded_dict_x['input_ids'],0)
    input_ids_y = np.squeeze(encoded_dict_y['input_ids'],0)
    # Mask of input without padding
    mask_x = np.squeeze(encoded_dict_x['attention_mask'],0)
    mask_y = np.squeeze(encoded_dict_y['attention_mask'],0)

    return {'ids_x': input_ids_x,
            'ids_y': input_ids_y,
            'mask_x': mask_x,
            'mask_y': mask_y,
            'labels': [label],}


class CommonlitDataset:
    def __init__(self, texts_x, texts_y, labels):
        self.texts_x = texts_x
        self.texts_y = texts_y
        self.labels = labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts_x)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.texts_x[item],
                            self.texts_y[item],
                            self.labels[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids_x': torch.tensor(data['ids_x'], dtype=torch.long),
                'ids_y': torch.tensor(data['ids_y'], dtype=torch.long),
                'mask_x': torch.tensor(data['mask_x'], dtype=torch.long),
                'mask_y': torch.tensor(data['mask_y'], dtype=torch.long),
                'labels': torch.tensor(data['labels'], dtype=torch.float),}
