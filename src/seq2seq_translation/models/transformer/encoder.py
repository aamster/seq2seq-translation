import torch
from torch import nn
from torch.nn import LayerNorm

from seq2seq_translation.models.transformer.multi_head_attention import MultiHeadSelfAttention
from seq2seq_translation.models.transformer.mlp import MLP, ActivationFunction
from seq2seq_translation.models.transformer._transformer import _Transformer
from seq2seq_translation.models.transformer.positional_encoding import PositionalEncodingType


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        norm_first: bool = False,
        mlp_activation: ActivationFunction = ActivationFunction.RELU
    ):
        super().__init__()
        self.layer_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])
        self.multi_head_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_attention_heads,
            dropout=dropout,
        )
        self.mlp = MLP(
            d_model=d_model, dropout=dropout, hidden_dim=feedforward_hidden_dim,
            activation_function=mlp_activation
        )
        self._norm_first = norm_first

    def forward(self, x, key_padding_mask: torch.tensor):
        if self._norm_first:
            x = self.layer_norm[0](x)
            x = x + self.multi_head_attention(x, key_padding_mask=key_padding_mask)
            x = x + self.mlp(self.layer_norm[1](x))
        else:
            x = x + self.multi_head_attention(x, key_padding_mask=key_padding_mask)
            x = self.layer_norm[0](x)
            x = x + self.mlp(x)
            x = self.layer_norm[1](x)
        return x

class EncoderTransformer(_Transformer):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        pad_token_idx: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        norm_first: bool = False,
        mlp_activation: ActivationFunction = ActivationFunction.RELU,
        positional_encoding_type: PositionalEncodingType = PositionalEncodingType.LEARNED
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            d_model=d_model,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            positional_encoding_type=positional_encoding_type,
            pad_token_idx=pad_token_idx
        )
        self.blocks = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model=d_model,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    norm_first=norm_first,
                    mlp_activation=mlp_activation
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.tensor, src_key_padding_mask: torch.tensor):
        x = self._calc_embeddings(x=x)
        for block in self.blocks:
            x = block(x, key_padding_mask=src_key_padding_mask)
        return x
