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

        #self.attention = SelfAttention()

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*9, config.HIDDEN_SIZE*6),
            torch.nn.GELU(),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*6, 1),
        )

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out_std = torch.std(out, dim=0)
        pooled_last_hidden_states = torch.cat((out_mean, out_max,out_std), dim=-1)
        
        #last_hidden_state = out.last_hidden_state

        ##Self attention
        #weights = self.attention(last_hidden_state, mask)
        #context_vector = torch.sum(weights * last_hidden_state, dim=1) 

        out_pooled_mean = torch.mean(pooled_last_hidden_states, dim=1)
        out_pooled_max, _ = torch.max(pooled_last_hidden_states, dim=1)
        out_pooled_std = torch.std(pooled_last_hidden_states, dim=1)
        context_vector = torch.cat([out_pooled_mean, out_pooled_max, out_pooled_std], dim=-1)

        return self.classifier(context_vector)
