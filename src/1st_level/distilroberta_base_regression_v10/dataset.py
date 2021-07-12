import numpy as np
import torch
import textstat

import config

def rescale_linear(x, minimum, maximum):
    """Rescale an arrary linearly."""
    m = 2 / (maximum - minimum)
    b = -1 - m * minimum
    return m * x + b

def process_data(text, label,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    sentences = text.split('.')
    sentences_encoded_dict = tokenizer.batch_encode_plus(
        sentences,                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = config.MAX_LEN,           # Pad & truncate all sentences.
        padding = 'max_length',
		return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation = True,
    )
    sentences_input_ids = sentences_encoded_dict['input_ids']
    sentences_mask = sentences_encoded_dict['attention_mask']
    sentences_features = [
                    rescale_linear(textstat.syllable_count(text), 0, 120),
                    rescale_linear(textstat.lexicon_count(text, removepunct=True), 0, 120),
                    ]

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

    document_features = [rescale_linear(textstat.flesch_reading_ease(text), -40, 120),
                        rescale_linear(textstat.smog_index(text), -5, 30),
                        rescale_linear(textstat.flesch_kincaid_grade(text), -5, 30),
                        rescale_linear(textstat.coleman_liau_index(text), -5, 30),
                        rescale_linear(textstat.automated_readability_index(text), -5, 30),
                        rescale_linear(textstat.dale_chall_readability_score(text), 0   , 12),
                        rescale_linear(textstat.difficult_words(text), 0, 70),
                        rescale_linear(textstat.linsear_write_formula(text), 0, 30),
                        rescale_linear(textstat.gunning_fog(text), 0, 30),
                        rescale_linear(textstat.fernandez_huerta(text), 0, 140),
                        rescale_linear(textstat.szigriszt_pazos(text), 0, 140),
                        rescale_linear(textstat.gutierrez_polini(text), 0, 60),
                        rescale_linear(textstat.crawford(text), -2, -7)]

    return {'sentences_ids': sentences_input_ids,
            'sentences_mask': sentences_mask,
            'sentences_features': sentences_features,
            'ids': input_ids,
            'mask': mask,
            'document_features': document_features,
            'labels': [label],
            }


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

        return {'sentences_ids': torch.tensor(data['sentences_ids'], dtype=torch.long),
                'sentences_mask': torch.tensor(data['sentences_mask'], dtype=torch.long),
                'sentences_features': torch.tensor(data['sentences_features'], dtype=torch.float),
                'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'document_features': torch.tensor(data['document_features'], dtype=torch.float),
                'labels': torch.tensor(data['labels'], dtype=torch.float),
                }
