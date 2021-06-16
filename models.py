#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig


class ElectraAnsTypHead(torch.nn.Module):
    def __init__(self):

        super(ElectraAnsTypHead, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.dense = torch.nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(self.electra.config.hidden_size, 3)
        self.gelu = torch.nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraQA(torch.nn.Module):
    def __init__(self):

        super(ElectraQA, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.qa_outputs = torch.nn.Linear(self.electra.config.hidden_size, 2)
        self.ansTyp_outputs = ElectraAnsTypHead()
        self.suppFacts_outputs = torch.nn.Linear(self.electra.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        ansTyp_logits = self.ansTyp_outputs(sequence_output)
        suppFacts_logits = self.suppFacts_outputs(sequence_output).squeeze(-1)

        return start_logits, end_logits, ansTyp_logits, suppFacts_logits

