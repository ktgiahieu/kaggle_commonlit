import torch
import transformers

import config

class SelfAttention(torch.nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.linear1 = torch.nn.Linear(config.HIDDEN_SIZE*2, config.ATTENTION_HIDDEN_SIZE*2)          
        self.tanh = torch.nn.Tanh()            
        self.linear2 = torch.nn.Linear(config.ATTENTION_HIDDEN_SIZE*2, 1)
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
            torch.nn.Linear(config.HIDDEN_SIZE*4, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, ids_x, ids_y, mask_x, mask_y):
        out_x = self.automodel(ids_x, attention_mask=mask_x)
        out_y = self.automodel(ids_y, attention_mask=mask_y)

        # Mean-max pooler
        #out_x = out_x.last_hidden_state
        out_x = torch.stack(
            tuple(out_x[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean_x = torch.mean(out_x, dim=0)
        out_max_x, _ = torch.max(out_x, dim=0)
        pooled_last_hidden_states_x = torch.cat((out_mean_x, out_max_x), dim=-1)
        

        out_y = torch.stack(
            tuple(out_y[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean_y = torch.mean(out_y, dim=0)
        out_max_y, _ = torch.max(out_y, dim=0)
        pooled_last_hidden_states_y = torch.cat((out_mean_y, out_max_y), dim=-1)


        #out_y = out_y.last_hidden_state

        weights_x = self.attention(pooled_last_hidden_states_x, mask_x)
        context_vector_x = torch.sum(weights_x * pooled_last_hidden_states_x, dim=1) 
        weights_y = self.attention(pooled_last_hidden_states_y, mask_y)
        context_vector_y = torch.sum(weights_y * pooled_last_hidden_states_y, dim=1) 

        context_vector = torch.cat([context_vector_x, context_vector_y], dim=-1)

        #out = torch.stack(
        #    tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        #out_mean = torch.mean(out, dim=0)
        #out_max, _ = torch.max(out, dim=0)
        #pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1)

        ##self attention
        #weights = self.attention(pooled_last_hidden_states, mask)
        #context_vector = torch.sum(weights * pooled_last_hidden_states, dim=1) 

        return self.classifier(context_vector)
