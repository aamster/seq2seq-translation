import math
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.va = nn.Linear(hidden_size, 1)

    def forward(self, s_t_minus_1, h_j):
        """

        :param s_t_minus_1: decoder previous hidden state
        :param h_j: encoder outputs at each timestep
        :return: attention weights
        """
        batch_size = h_j.shape[0]

        s_t_minus_1 = s_t_minus_1.reshape(batch_size, 1, -1)
        scores = self.va(torch.tanh(self.Wa(s_t_minus_1) + self.Ua(h_j)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)

        return weights, scores


class CosineSimilarityAttention(nn.Module):
    def __init__(
        self,
        encoder_output_size: int,
        query_dim: int,
        Dv: int,
        dropout: float = 0.0
    ):
        super().__init__()
        Dx = encoder_output_size
        Dq = query_dim
        self.Wk = nn.Linear(Dx, Dq)
        self.Wv = nn.Linear(Dx, Dv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, x, mask: torch.tensor):
        """

        :param query: decoder hidden state
        :param x: encoder output at each timestep
        :param mask: ignore the pad tokens in the encoder output
        :return: attention weights
        """
        keys = self.Wk(x)
        values = self.Wv(x)

        query = query[-1].unsqueeze(0)    # take only the last hidden layer
        Dq = query.shape[-1]
        query = query.squeeze(0).unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(Dq)

        # mask the pad token
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        Y = attention.bmm(values)
        return Y, attention


class AttentionType(Enum):
    CosineSimilarityAttention = 'CosineSimilarityAttention'
    BahdanauAttention = 'BahdanauAttention'
