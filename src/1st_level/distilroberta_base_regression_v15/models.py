import torch
import transformers

import config

class SelfAttention(torch.nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, config.ATTENTION_HIDDEN_SIZE)          
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

        self.attention_sent1 = SelfAttention(config.HIDDEN_SIZE*2)
        self.attention_sent2 = SelfAttention(config.HIDDEN_SIZE*2)
        self.attention_doc = SelfAttention(config.HIDDEN_SIZE*2)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*4 , config.HIDDEN_SIZE*4),
            torch.nn.Dropout(config.CLASSIFIER_DROPOUT),
            torch.nn.Linear(config.HIDDEN_SIZE*4, 1),
        )

        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, ids, mask, document_features,
                        sentences_ids, sentences_mask, sentences_features, sentences_attention_mask):

        sentences_vector = []
        #iterate through each excerpt, sent_ids is of shape (MAX_N_SENTENCE, MAX_LEN_SENTENCE)
        #sent_out.last_hidden_state is of shape (MAX_N_SENTENCE, MAX_LEN_SENTENCE, HIDDEN_SIZE)
        for sent_ids, sent_mask, sent_features, sent_attention_mask in zip(sentences_ids, sentences_mask, sentences_features, sentences_attention_mask):
            sent_out = self.automodel(sent_ids, attention_mask=sent_mask)

            # Mean-max pooler
            sent_out = sent_out.hidden_states
            sent_out = torch.stack(
                tuple(sent_out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
            sent_out_mean = torch.mean(sent_out, dim=0)
            sent_out_max, _ = torch.max(sent_out, dim=0)
            sent_pooled_last_hidden_states = torch.cat((sent_out_mean, sent_out_max), dim=-1)

            #Self attention
            weights1 = self.attention_sent1(sent_pooled_last_hidden_states, sent_mask)
            sent_context_vector = torch.sum(weights1 * sent_pooled_last_hidden_states, dim=1) 

            #concat external features
            #context_and_sentence_vector = torch.cat((sent_context_vector, sent_features), dim=-1)

            #Self attention
            weights2 = self.attention_sent2(sent_context_vector, sent_attention_mask)
            sent_attentioned_vector = torch.sum(weights2 * sent_context_vector, dim=0) 
            sentences_vector.append(sent_attentioned_vector)
            
        sentences_vector = torch.stack(sentences_vector, dim=0)

        #The hold excerpt
        out = self.automodel(ids, attention_mask=mask)

        # Mean-max pooler
        out = out.hidden_states
        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        pooled_last_hidden_states = torch.cat((out_mean, out_max), dim=-1)

        #Self attention
        weights = self.attention_doc(pooled_last_hidden_states, mask)
        context_vector = torch.sum(weights * pooled_last_hidden_states, dim=1) 

        #Multisample-Dropout
        ##

        #Add Document level features, sentence level features
        feature_vector = torch.cat((context_vector, sentences_vector), dim=-1)

        return self.classifier(feature_vector)
