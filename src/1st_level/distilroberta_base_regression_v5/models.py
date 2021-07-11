import torch
import transformers

import config


class CommonlitModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(CommonlitModel, self).__init__(conf)
        self.automodel = transformers.AutoModel.from_pretrained(
            config.MODEL_CONFIG,
            config=conf)

        self.attention = torch.nn.Sequential(            
            torch.nn.Linear(config.HIDDEN_SIZE*2, config.ATTENTION_HIDDEN_SIZE),            
            torch.nn.Tanh(),                       
            torch.nn.Linear(config.ATTENTION_HIDDEN_SIZE, 1),
            torch.nn.Softmax(dim=1)
        )   

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*2, 1),
        )
        
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, ids, mask):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1)

        #Self attention
        weights = self.attention(pooled_last_hidden_states)

        context_vector = torch.sum(weights * pooled_last_hidden_states, dim=1) 

        #Multisample-Dropout
        ##

        return self.classifier(context_vector)
