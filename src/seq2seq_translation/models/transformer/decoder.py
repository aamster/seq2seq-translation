from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm

from seq2seq_translation.models.transformer.multi_head_attention import MultiHeadSelfAttention, \
    MultiHeadCrossAttention
from seq2seq_translation.models.transformer._transformer import _Transformer
from seq2seq_translation.models.transformer.mlp import MLP, ActivationFunction


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        feedforward_hidden_dim: int = 2048,
        norm_first: bool = False,
        mlp_activation: ActivationFunction = ActivationFunction.RELU
    ):
        super().__init__()
        self.masked_multi_head_self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_attention_heads,
            dropout=dropout,
        )
        layer_norms = [
            LayerNorm(d_model) for _ in range(3 if use_cross_attention else 2)
        ]
        if use_cross_attention:
            self.multi_head_cross_attention = MultiHeadCrossAttention(
                d_model=d_model,
                n_head=n_attention_heads,
                dropout=dropout,
            )

        self.mlp = MLP(
            d_model=d_model, dropout=dropout, hidden_dim=feedforward_hidden_dim,
            activation_function=mlp_activation
        )
        self.layer_norm = nn.ModuleList(layer_norms)
        self._use_cross_attention = use_cross_attention
        self._norm_first = norm_first

    def forward(
        self,
        x: torch.tensor,
        tgt_key_padding_mask: torch.tensor,
        memory_key_padding_mask: Optional[torch.tensor] = None,
        memory: Optional[torch.tensor] = None,
    ):
        if self._use_cross_attention:
            if memory is None:
                raise ValueError("must provide memory to use cross attention")

        if self._norm_first:
            x = self.layer_norm[0](x)
            x = x + self.masked_multi_head_self_attention(
                x, key_padding_mask=tgt_key_padding_mask, is_causal=True
            )
            if self._use_cross_attention:
                x = self.layer_norm[1](x)
                x = x + self.multi_head_cross_attention(
                    query=x,
                    key=memory,
                    key_padding_mask=memory_key_padding_mask,
                )
            x = self.layer_norm[2 if self._use_cross_attention else 1](x)
            x = x + self.mlp(x)
        else:
            x = x + self.masked_multi_head_self_attention(
                x, key_padding_mask=tgt_key_padding_mask, is_causal=True
            )
            x = self.layer_norm[0](x)
            if self._use_cross_attention:
                x = x + self.multi_head_cross_attention(
                    query=x,
                    key=memory,
                    key_padding_mask=memory_key_padding_mask,
                )
                x = self.layer_norm[1](x)
            x = x + self.mlp(x)
            x = self.layer_norm[2](x)
        return x


class DecoderTransformer(_Transformer):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        feedforward_hidden_dim: int = 2048,
        norm_first: bool = False,
        mlp_activation: ActivationFunction = ActivationFunction.RELU
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
        )
        self._use_cross_attention = use_cross_attention
        self.blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    d_model=d_model,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    use_cross_attention=use_cross_attention,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    norm_first=norm_first,
                    mlp_activation=mlp_activation
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(self._d_model, self._vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.embedding.weight = self.lm_head.weight

        self.layer_norm = LayerNorm(self._d_model)

    def forward(
        self,
        x: torch.tensor,
        memory_key_padding_mask: Optional[torch.tensor] = None,
        tgt_key_padding_mask: Optional[torch.tensor] = None,
        memory: Optional[torch.tensor] = None,
    ):
        if self._use_cross_attention and memory is None:
            raise ValueError("must provide memory if use_cross_attention")

        x = self._calc_embeddings(x=x)
        for block in self.blocks:
            x = block(
                x=x,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits