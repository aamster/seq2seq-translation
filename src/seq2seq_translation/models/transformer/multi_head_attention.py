import os
from typing import Optional

import torch
from torch import nn


class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self._d_model = d_model
        self.n_head = n_head
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.dropout = dropout


    @property
    def d_qkv(self) -> int:
        return int(self._d_model / self.n_head)

    def forward(self, **kwargs):
        raise NotImplementedError

    def _calc_attention(self, q, k, v, attn_mask: Optional[torch.tensor] = None):
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        q = q.view(B, T_q, self.n_head, self.d_qkv).transpose(1, 2)
        k = k.view(B, T_k, self.n_head, self.d_qkv).transpose(1, 2)
        v = v.view(B, T_k, self.n_head, self.d_qkv).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T_q, self._d_model)

        y = self.proj_dropout(self.output_proj(y))

        return y

    @staticmethod
    def _create_attn_mask(
            key_padding_mask: Optional[torch.tensor], T_q: int, T_k: int, is_causal: bool
    ):
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_k)
        else:
            attn_mask = torch.ones((1, 1, 1, T_k), device=os.environ['DEVICE'], dtype=torch.bool)

        if is_causal:
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=attn_mask.device, dtype=torch.bool))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T_q, T_k)
            attn_mask = causal_mask & attn_mask

        return attn_mask

class MultiHeadSelfAttention(_MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout,
        )
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(
        self,
        x: torch.tensor,
        is_causal: bool = False,
        key_padding_mask: Optional[torch.tensor] = None
    ):
        # calculates q, k, v in a single operation rather than in 3 separate operations for
        # efficiency but is equivalent
        q, k, v = self.qkv_proj(x).split(self._d_model, dim=2)

        attn_mask = self._create_attn_mask(
            key_padding_mask=key_padding_mask,
            T_k=k.shape[1],
            T_q=q.shape[1],
            is_causal=is_causal
        )

        y = self._calc_attention(
            q=q, k=k, v=v, attn_mask=attn_mask
        )

        return y


class MultiHeadCrossAttention(_MultiHeadAttention):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout,
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        is_causal: bool = False,
        key_padding_mask: Optional[torch.tensor] = None
    ):
        q = self.q_proj(query)
        k, v = self.kv_proj(key).split(self._d_model, dim=2)

        attn_mask = self._create_attn_mask(
            key_padding_mask=key_padding_mask,
            T_k=k.shape[1],
            T_q=q.shape[1],
            is_causal=is_causal
        )
        y = self._calc_attention(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
        )
        return y