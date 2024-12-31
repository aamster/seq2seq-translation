import torch
from torch import nn as nn

from seq2seq_translation.models.transformer.positional_encoding import PositionalEncodingType


class _Transformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.0,
        positional_encoding_type: PositionalEncodingType = PositionalEncodingType.LEARNED
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
        self._positional_encoding_type = positional_encoding_type

    def _calc_embeddings(self, x: torch.tensor):
        device = x.device
        b, t = x.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.embedding(x)  # token embeddings of shape (b, t, d_model)

        if self._positional_encoding_type == PositionalEncodingType.LEARNED:
            assert (
                t <= self._block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self._block_size}"
            pos_emb = self.positional_encoding(pos)  # (t, d_model)
            x = tok_emb + pos_emb
        elif self._positional_encoding_type == PositionalEncodingType.SINUSOIDAL:
            pos[::2] = torch.sin(pos[::2] / 1e4**(2*pos[::2]/self._d_model))
            pos[1::2] = torch.cos(pos[1::2] / 1e4 ** (2 * pos[1::2] / self._d_model))
            x = tok_emb + pos
        else:
            raise ValueError(f'{self._positional_encoding_type} not supported')
        x = self.dropout(x)
        return x
