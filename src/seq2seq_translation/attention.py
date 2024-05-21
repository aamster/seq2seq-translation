import math
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, encoder_bidirectional: bool = False):
        super().__init__()
        # D = 2 if encoder_bidirectional else 1
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, 1)

    def forward(self, query, x):
        # query is decoder hidden state
        # inputs x are encoder output at each timestep

        batch_size = x.shape[0]

        query = query.reshape(batch_size, 1, -1)
        scores = self.Wv(torch.tanh(self.Wq(query) + self.Wk(x)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)

        return weights


class CosineSimilarityAttention(nn.Module):
    def __init__(
        self,
        encoder_output_size: int,
        decoder_hidden_size: int,
        Dv: int
    ):
        super().__init__()
        Dx = encoder_output_size
        Dq = decoder_hidden_size
        self.Wk = nn.Linear(Dx, Dq)
        self.Wv = nn.Linear(Dx, Dv)

    def forward(self, query, x):
        """

        :param query: decoder hidden state
        :param x: encoder output at each timestep
        :return: attention weights
        """
        keys = self.Wk(x)
        values = self.Wv(x)

        Dq = query.shape[-1]
        scores = query.permute(1, 0, 2).bmm(keys.permute(0, 2, 1)) / math.sqrt(Dq)
        attention = F.softmax(scores, dim=-1)
        Y = attention.bmm(values)
        return Y


class AttentionType(Enum):
    CosineSimilarityAttention = 'CosineSimilarityAttention'
    BahdanauAttention = 'BahdanauAttention'
