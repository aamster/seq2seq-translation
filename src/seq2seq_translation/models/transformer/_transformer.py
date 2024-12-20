import torch
from torch import nn as nn


class _Transformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._block_size = block_size
        self._d_model = d_model
        self._dropout = dropout
        self._n_attention_heads = n_attention_heads
        self._n_layers = n_layers

        self.embedding = nn.Embedding(self._vocab_size, self._d_model)
        self.positional_encoding = nn.Embedding(self._block_size, self._d_model)
        self.dropout = nn.Dropout(self._dropout)

    def _calc_embeddings(self, x: torch.tensor):
        device = x.device
        b, t = x.size()
        assert (
            t <= self._block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self._block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.embedding(x)  # token embeddings of shape (b, t, d_model)
        pos_emb = self.positional_encoding(pos)  # (t, d_model)
        x = self.dropout(tok_emb + pos_emb)
        return x
