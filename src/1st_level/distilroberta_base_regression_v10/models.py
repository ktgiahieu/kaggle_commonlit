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

        #iterate through each excerpt, sent_ids is of shape (MAX_N_SENTENCE, MAX_LEN_SENTENCE)
        #sent_out.last_hidden_state is of shape (MAX_N_SENTENCE, MAX_LEN_SENTENCE, HIDDEN_SIZE)
        sentences_vector = []
        for sent_ids, sent_mask, sent_features in zip(sentences_ids, sentences_mask, sentences_features):
            print(sent_ids.shape, sent_mask.shape, sent_features.shape)
            sent_out = self.automodel(sent_ids, attention_mask=sent_mask)
            sent_last_hidden_state = sent_out.last_hidden_state
            sent_context_vector = sent_last_hidden_state[:,0,:]
            context_and_sentence_vector = torch.cat((sent_context_vector, sent_features), dim=-1)
            sentences_vector.append(torch.mean(context_and_sentence_vector, dim=0))
            ## improve with masked
        sentences_vector = torch.tensor(sentences_vector, dtype=torch.float)
        print(sentences_vector.shape)

        out = self.automodel(ids, attention_mask=mask)
        last_hidden_state = out.last_hidden_state

        context_vector = last_hidden_state[:,0,:]

        #Multisample-Dropout
        ##

        #Add Document level features
        context_and_document_vector = torch.cat((context_vector, document_features), dim=-1)


        return self.classifier(context_and_document_vector)
