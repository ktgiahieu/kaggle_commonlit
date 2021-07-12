import torch
import transformers

import config

class SelfAttention(torch.nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.linear1 = torch.nn.Linear(config.HIDDEN_SIZE, config.ATTENTION_HIDDEN_SIZE)          
        self.tanh = torch.nn.Tanh()            
        self.linear2 = torch.nn.Linear(config.ATTENTION_HIDDEN_SIZE, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def masked_vector(self, vector, mask):
        return vector + (mask.unsqueeze(-1) + 1e-45).log()

    def forward(self, x, mask):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.masked_vector(out, mask)
        out = self.softmax(out)
        return out

class CommonlitModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(CommonlitModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        self.attention = SelfAttention()

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE + config.N_DOCUMENT_FEATURES, 512),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(512, 1),
        )

        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, ids, mask, document_features,
                        sentences_ids, sentences_mask, sentences_features):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        print(sentences_ids.shape, sentences_mask.shape, sentences_features.shape, )
        out = self.automodel(ids, attention_mask=mask)
        last_hidden_state = out.last_hidden_state

        context_vector = last_hidden_state[:,0,:]

        #Multisample-Dropout
        ##

        #Add Document level features
        context_and_document_vector = torch.cat((context_vector, document_features), dim=-1)

        return self.classifier(context_and_document_vector)
