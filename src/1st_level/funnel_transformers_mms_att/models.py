import torch
import transformers

import config

class SelfAttention(torch.nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.linear1 = torch.nn.Linear(config.HIDDEN_SIZE*3, config.ATTENTION_HIDDEN_SIZE*2)          
        self.tanh = torch.nn.Tanh()            
        self.linear2 = torch.nn.Linear(config.ATTENTION_HIDDEN_SIZE*2, 1)
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

        self.attention_decoder = SelfAttention()
        self.attention_block3 = SelfAttention()

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*6, config.HIDDEN_SIZE*4),
            torch.nn.GELU(),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*4, 1),
        )

    def forward(self, ids, mask):
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        #print([out[-i - 1].shape for i in range(26)])
        #Decoder Block
        #out_decoder = torch.stack(
        #    tuple(out[-i - 1] for i in range(3)), dim=0)
        #out_mean_decoder = torch.mean(out_decoder, dim=0)
        #out_max_decoder, _ = torch.max(out_decoder, dim=0)
        #out_std_decoder = torch.std(out_decoder, dim=0)
        #pooled_last_hidden_states_decoder = torch.cat((out_mean_decoder, out_max_decoder, out_std_decoder), dim=-1)
        ##Self attention
        #weights_decoder = self.attention_decoder(pooled_last_hidden_states_decoder, mask)
        #context_vector_decoder = torch.sum(weights_decoder * pooled_last_hidden_states_decoder, dim=1) 

        #Encoder Block 3
        out_block3 = torch.stack(
            tuple(out[-i - 1] for i in range(3,11)), dim=0)
        out_mean_block3 = torch.mean(out_block3, dim=0)
        out_max_block3, _ = torch.max(out_block3, dim=0)
        out_std_block3 = torch.std(out_block3, dim=0)
        pooled_last_hidden_states_block3 = torch.cat((out_mean_block3, out_max_block3, out_std_block3), dim=-1)
        #Self attention
        weights_block3 = self.attention_block3(pooled_last_hidden_states_block3)
        context_vector_block3 = torch.sum(weights_block3 * pooled_last_hidden_states_block3, dim=1) 

        #context_vector = torch.cat([context_vector_block3, context_vector_decoder], dim=-1) 
        context_vector = context_vector_block3

        return self.classifier(context_vector)
