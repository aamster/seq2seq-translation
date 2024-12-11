from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention, Transformer

from seq2seq_translation.models.transformer.multi_head_attention import MultiHeadSelfAttention, \
    MultiHeadCrossAttention
from seq2seq_translation.models.transformer._transformer import _Transformer
from seq2seq_translation.models.transformer.mlp import MLP


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        feedforward_hidden_dim: int = 2048,
    ):
        super().__init__()
        # self.masked_multi_head_self_attention = MultiHeadSelfAttention(
        #     d_model=d_model,
        #     n_head=n_attention_heads,
        #     is_causal=True,
        #     dropout=dropout,
        # )
        self.masked_multi_head_self_attention = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_attention_heads,
            dropout=dropout,
            bias=False,
            batch_first=True
        )
        layer_norms = [
            LayerNorm(d_model) for _ in range(4 if use_cross_attention else 2)
        ]
        if use_cross_attention:
            # self.multi_head_cross_attention = MultiHeadCrossAttention(
            #     d_model=d_model,
            #     n_head=n_attention_heads,
            #     dropout=dropout,
            # )
            self.multi_head_cross_attention = MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_attention_heads,
                dropout=dropout,
                bias=False,
                batch_first=True
            )

        self.mlp = MLP(
            d_model=d_model, dropout=dropout, hidden_dim=feedforward_hidden_dim
        )
        self.layer_norm = nn.ModuleList(layer_norms)
        self._use_cross_attention = use_cross_attention

    def forward(
        self,
        x: torch.tensor,
        input_pad_mask: torch.tensor,
        output_pad_mask: Optional[torch.tensor] = None,
        encoder_out: Optional[torch.tensor] = None,
        need_attn_weights: bool = False
    ):
        if self._use_cross_attention:
            if encoder_out is None:
                raise ValueError("must provide encoder_out to use cross attention")
        # x = x + self.masked_multi_head_self_attention(
        #     self.layer_norm[0](x), pad_mask=output_pad_mask
        # )
        x = self.layer_norm[0](x)
        causal_mask = Transformer.generate_square_subsequent_mask(
            sz=x.shape[1],
            device=x.device,
            dtype=x.dtype
        )
        attn_output, attn_output_weights = self.masked_multi_head_self_attention(
            query=x, key=x, value=x, is_causal=True,
            need_weights=need_attn_weights,
            attn_mask=causal_mask
        )
        x = x + attn_output
        if self._use_cross_attention:
            x = self.layer_norm[1](x)
            encoder_out = self.layer_norm[2](encoder_out)
            attn_output, attn_output_weights = self.multi_head_cross_attention(
                query=x, key=encoder_out, value=encoder_out, key_padding_mask=~input_pad_mask,
                need_weights=need_attn_weights
            )
            x = x + attn_output
            # x = x + self.multi_head_cross_attention(
            #     query=self.layer_norm[1](x),
            #     key=self.layer_norm[2](encoder_out),
            #     query_pad_mask=output_pad_mask,
            #     key_pad_mask=input_pad_mask,
            # )
        x = x + self.mlp(self.layer_norm[3 if self._use_cross_attention else 1](x))
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
        input_pad_mask: torch.tensor,
        output_pad_mask: Optional[torch.tensor] = None,
        encoder_out: Optional[torch.tensor] = None,
    ):
        if self._use_cross_attention and encoder_out is None:
            raise ValueError("must provide encoder_out if use_cross_attention")

        x = self._calc_embeddings(x=x)
        for block in self.blocks:
            x = block(
                x=x,
                encoder_out=encoder_out,
                input_pad_mask=input_pad_mask,
                output_pad_mask=output_pad_mask,
            )
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits