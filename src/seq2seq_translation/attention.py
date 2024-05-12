import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, encoder_bidirectional: bool = False):
        super(BahdanauAttention, self).__init__()

        D = 2 if encoder_bidirectional else 1
        self.Wa = nn.Linear(D * hidden_size, hidden_size)
        self.Ua = nn.Linear(D * hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query is decoder hidden state
        # keys are encoder output at each timestep

        batch_size = keys.shape[0]

        query = query.reshape(batch_size, 1, -1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)

        return weights