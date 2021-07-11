import numpy as np
import torch
import textstat

import config

def process_data(text, label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    encoded_dict = tokenizer.encode_plus(
        text,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        padding = 'max_length',
		return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = True,
    )
    # ----------------------------------

    # Input for BERT
    input_ids = np.squeeze(encoded_dict['input_ids'],0)
    # Mask of input without padding
    mask = np.squeeze(encoded_dict['attention_mask'],0)

    document_features = [textstat.flesch_reading_ease(text),
                        textstat.smog_index(text),
                        textstat.flesch_kincaid_grade(text),
                        textstat.coleman_liau_index(text),
                        textstat.automated_readability_index(text),
                        textstat.dale_chall_readability_score(text),
                        textstat.difficult_words(text),
                        textstat.linsear_write_formula(text),
                        textstat.gunning_fog(text),
                        textstat.fernandez_huerta(text),
                        textstat.szigriszt_pazos(text),
                        textstat.gutierrez_polini(text),
                        textstat.crawford(text)]

    return {'ids': input_ids,
            'mask': mask,
            'labels': [label],
            'document_features': document_features}


class CommonlitDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.texts[item],
                            self.labels[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'labels': torch.tensor(data['labels'], dtype=torch.float),
                'document_features': torch.tensor(data['document_features'], dtype=torch.float),}
