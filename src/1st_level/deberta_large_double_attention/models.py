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

    def forward(self, x, mask=None):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)

        if mask is not None:
            out = self.masked_vector(out, mask)
        out = self.softmax(out)
        return out

class CommonlitModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(CommonlitModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        self.attention_pooler = SelfAttention()

        self.attention_head = SelfAttention()

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE, 1),
        )

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Attention pooler
        out = out.hidden_states #24 arrays: (8, 248, 768)
        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=2) #(8, 248, 4, 768)

        pooler_weights = self.attention_pooler(out)
        pooled_last_hidden_states = torch.sum(pooler_weights * out, dim=2) #(8, 248, 768)

        #Self attention

        head_weights = self.attention_head(pooled_last_hidden_states, mask)
        context_vector = torch.sum(head_weights * pooled_last_hidden_states, dim=1) #(8, 768)

        return self.classifier(context_vector) #(8)
