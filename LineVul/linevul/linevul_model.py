import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, extra_dim):
        super().__init__()
        # TODO: add dim on just dense or both?
        self.dense = nn.Linear(config.hidden_size + extra_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, flowgnn_embed, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if flowgnn_embed is not None:
            x = torch.cat((x, flowgnn_embed), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, flowgnn_encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        if not args.no_flowgnn:
            self.flowgnn_encoder = flowgnn_encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config, 0 if args.no_flowgnn else self.flowgnn_encoder.out_dim)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, graphs=None, output_attentions=False, input_ids=None):
        if self.args.no_flowgnn:
            flowgnn_embed = None
        elif graphs is not None:
            flowgnn_embed = self.flowgnn_encoder(graphs, {})
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state, flowgnn_embed)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs, flowgnn_embed)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob


class DeepDFAModel(RobertaForSequenceClassification):   
    def __init__(self, flowgnn_encoder, config):
        super(DeepDFAModel, self).__init__(config=config)
        self.encoder = flowgnn_encoder
        self.dense = nn.Linear(self.encoder.out_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, input_embed=None, labels=None, graphs=None, output_attentions=False, input_ids=None):
        flowgnn_embed = self.encoder(graphs, {})
        feats = self.dropout(flowgnn_embed)
        feats = self.dense(feats)
        feats = torch.tanh(feats)
        feats = self.dropout(feats)
        feats = self.out_proj(feats)
        prob = torch.softmax(feats, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(feats, labels)
            return loss, prob
        else:
            return prob
