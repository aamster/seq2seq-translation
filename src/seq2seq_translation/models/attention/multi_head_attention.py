import os
from typing import Optional

import torch
from torch import nn


class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        is_causal: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self._d_model = d_model
        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.is_causal = is_causal


    @property
    def d_qkv(self) -> int:
        return int(self._d_model / self.n_head)

    def forward(self, **kwargs):
        raise NotImplementedError

    def _calc_attention(self, q, k, v, query_pad_mask: Optional[torch.tensor], key_pad_mask: Optional[torch.tensor]):
        """
        Compute attention using scaled_dot_product_attention, skipping fully padded queries.
        """
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        # Create the attention mask
        attn_mask = self._create_attn_mask(
            key_pad_mask=key_pad_mask, T_q=T_q, T_k=T_k
        )

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        if query_pad_mask is not None:
            # Ensure outputs for padded queries are zeroed out
            y[~query_pad_mask] = 0.0

        return y

    def _create_attn_mask(
            self, key_pad_mask: Optional[torch.tensor], T_q, T_k
    ):
        """
        Create an attention mask combining query, key padding, and optional causal masks.
        """
        if key_pad_mask is not None:
            attn_mask = key_pad_mask.unsqueeze(1)  # (B, 1, T_k)
        else:
            attn_mask = torch.ones((1, 1, T_k), device=os.environ['DEVICE'], dtype=torch.bool)

        if self.is_causal:
            causal_mask = torch.tril(torch.ones(T_q, T_k, dtype=torch.bool, device=attn_mask.device))
            causal_mask = causal_mask.unsqueeze(0)  # (1, T_q, T_k)
            attn_mask = causal_mask & attn_mask

        float_mask = torch.where(
            attn_mask,
            torch.tensor(0.0, device=attn_mask.device),
            torch.tensor(float('-inf'), device=attn_mask.device),
        )

        return float_mask

class MultiHeadSelfAttention(_MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        is_causal: bool,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            is_causal=is_causal,
            dropout=dropout,
        )

    def forward(self, x: torch.tensor, pad_mask: torch.tensor):
        # calculates q, k, v in a single operation rather than in 3 separate operations for
        # efficiency but is equivalent
        q, k, v = self.qkv_proj(x).split(self._d_model, dim=2)

        y = self._calc_attention(
            q=q, k=k, v=v, query_pad_mask=pad_mask, key_pad_mask=pad_mask
        )
        y = self.proj_dropout(self.output_proj(y))

        return y


class MultiHeadCrossAttention(_MultiHeadAttention):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            is_causal=False,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        query_pad_mask: torch.tensor,
        key_pad_mask: torch.tensor,
    ):
        B, T_query, C = query.size()
        _, T_key, _ = key.size()

        combined = torch.cat([query, key], dim=1)  # (B, T_query + T_key, d_model)
        q, k, v = self.qkv_proj(combined).split(self._d_model, dim=2)

        q = q[:, :T_query]  # (B, T_query, attention_dim)
        k = k[:, T_query:]  # (B, T_key, attention_dim)
        v = v[:, T_query:]  # (B, T_key, attention_dim)

        y = self._calc_attention(
            q=q,
            k=k,
            v=v,
            query_pad_mask=query_pad_mask,
            key_pad_mask=key_pad_mask,
        )
        y = self.proj_dropout(self.output_proj(y))
        return y