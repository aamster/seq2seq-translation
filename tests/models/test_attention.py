import math

import pytest
import torch

from seq2seq_translation.models.attention.multi_head_attention import _MultiHeadAttention


class TestMultiHeadAttention:
    @pytest.mark.parametrize(
        'key_pad_mask, is_causal, expected', [
            (
                torch.tensor([[True, True, False]]),
                False,
                torch.tensor([[[[0.0, 0.0, -float('inf')]]]])
            ),
            (
                    torch.tensor([[True, True, False, False, False]]),
                    False,
                    torch.tensor([[[[0.0, 0.0, -float('inf'), -float('inf'), -float('inf')]]]])
            ),
            (
                    torch.tensor([[True, True, True]]),
                    False,
                    torch.tensor([[[[0.0, 0.0, 0.0]]]])
            ),
            (
                torch.tensor([[True, True, False]]),
                True,
                torch.tensor(
                    [
                        [
                            [0.0, -float('inf'), -float('inf')],
                            [0.0, 0.0, -float('inf')],
                            [0.0, 0.0, -float('inf')],
                            [0.0, 0.0, -float('inf')]
                        ]
                    ]
                )
            ),
            (
                    torch.tensor([[True, True, True]]),
                    True,
                    torch.tensor(
                        [
                            [
                                [0.0, -float('inf'), -float('inf')],
                                [0.0, 0.0, -float('inf')],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]
                            ]
                        ]
                    )
            ),
            (
                    torch.tensor([[True, True, True, False, False]]),
                    True,
                    torch.tensor(
                        [
                            [
                                [0.0, -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                                [0.0, 0.0, -float('inf'), -float('inf'), -float('inf')],
                                [0.0, 0.0, 0.0, -float('inf'), -float('inf')],
                                [0.0, 0.0, 0.0, -float('inf'), -float('inf')]
                            ]
                        ]
                    )
            )
        ]
    )
    def test__create_attn_mask(self, key_pad_mask, is_causal, expected):
        a = _MultiHeadAttention(
            n_embd=2,
            n_head=1,
            qkv_dim=8,
            is_causal=is_causal
        )
        attn_mask = a._create_attn_mask(key_padding_mask=key_pad_mask, T_k=key_pad_mask.shape[1], T_q=4)
        assert (attn_mask == expected).all()
