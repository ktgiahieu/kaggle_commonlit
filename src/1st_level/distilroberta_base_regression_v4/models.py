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
            torch.nn.Linear(config.HIDDEN_SIZE, config.ATTENTION_HIDDEN_SIZE),            
            torch.nn.Tanh(),                       
            torch.nn.Linear(config.ATTENTION_HIDDEN_SIZE, 1),
            torch.nn.Softmax(dim=1)
        )   

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE, 1),
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
        last_hidden = out.last_hidden_state

        #Self attention
        weights = self.attention(last_hidden)

        context_vector = torch.sum(weights * last_hidden, dim=1) 

        return self.classifier(context_vector)
