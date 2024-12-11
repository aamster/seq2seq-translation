import torch
from torch import nn
from torch.nn import LayerNorm

from seq2seq_translation.models.transformer.multi_head_attention import MultiHeadSelfAttention
from seq2seq_translation.models.transformer.mlp import MLP
from seq2seq_translation.models.transformer._transformer import _Transformer


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.layer_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])
        self.multi_head_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_attention_heads,
            is_causal=False,
            dropout=dropout,
        )
        self.mlp = MLP(
            d_model=d_model, dropout=dropout, hidden_dim=feedforward_hidden_dim
        )

    def forward(self, x, pad_mask: torch.tensor):
        x = x + self.multi_head_attention(self.layer_norm[0](x), pad_mask=pad_mask)
        x = x + self.mlp(self.layer_norm[1](x))
        return x

class EncoderTransformer(_Transformer):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            d_model=d_model,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model=d_model,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.tensor, pad_mask: torch.tensor):
        x = self._calc_embeddings(x=x)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        return x
