import torch
import torch.nn as nn
import math

class LSTM(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, max_len=60, dropout=0.1,
                 partition_init=150.0,
                 **kwargs):
        super().__init__()
        self.input = nn.Embedding(num_tokens+1, num_hid)
        self.hidden = nn.LSTM(
            num_hid,
            num_hid,
            batch_first=True,
            num_layers=num_layers,
        )

        self.output = nn.Linear(num_hid, num_tokens)
        self.num_tokens = num_tokens
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.input.parameters()) + list(self.hidden.parameters()) + list(self.output.parameters())

    def Z_param(self):
        return [self._Z]

    def forward(self, x, mask, return_all=False, lens=None,hidden=None):
        if return_all:
        
            out = self.input(x)
            out, _ = self.hidden(out, None)
            out = self.output(out)
            return out
        out = self.input(x)
        out, hidden = self.hidden(out,hidden)
        logit = self.output(out)
        # logit[:,:,-1] = -math.inf
        return logit, hidden

