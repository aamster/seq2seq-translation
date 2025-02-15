import torch
from torch import nn as nn

from seq2seq_translation.models.transformer.positional_encoding import PositionalEncodingType


class EmbeddingWithPadding(nn.Embedding):
    def __init__(self, num_embeddings: int, d_model: int, pad_idx: int = 50257):
        """
        Custom embedding layer that uses a standard embedding lookup for indices
        0 to num_embeddings-1, but when the index equals pad_idx (50257 by default),
        returns a zero vector of size d_model.

        Args:
            num_embeddings (int): Number of embeddings, e.g. 50257.
            d_model (int): Dimension of each embedding vector.
            pad_idx (int): The index that should be treated as padding. For inputs
                           equal to pad_idx, a zero vector is returned.
        """
        super().__init__(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.pad_idx = pad_idx

    def calc_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Looks up embeddings for the given indices. For any element equal to pad_idx,
        returns a zero vector.

        Args:
            indices (torch.Tensor): Tensor of token indices of any shape. Note that
                                    valid indices for lookup are 0 to num_embeddings-1.
                                    This method supports indices equal to pad_idx,
                                    even though pad_idx is outside the usual range.

        Returns:
            torch.Tensor: Tensor of embeddings with shape (*indices.shape, d_model).
        """
        # Create a boolean mask for pad tokens.
        pad_mask = (indices == self.pad_idx)

        # To avoid an out-of-bound error during the lookup,
        # replace pad_idx values with a safe index (e.g. 0).
        safe_indices = indices.clone()
        safe_indices[pad_mask] = 0  # 0 is guaranteed to be in-bound.

        # Do the standard embedding lookup.
        output = self(safe_indices)

        # For positions where the original index was pad_idx, zero out the output.
        # We need to unsqueeze the mask to match the embedding's last dimension.
        output = torch.where(pad_mask.unsqueeze(-1), torch.zeros_like(output), output)

        return output

class _Transformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        pad_token_idx: int,
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

        self.embedding = EmbeddingWithPadding(num_embeddings=vocab_size, d_model=d_model, pad_idx=pad_token_idx)
        if positional_encoding_type == PositionalEncodingType.LEARNED:
            self.positional_embedding = nn.Embedding(self._block_size, self._d_model)
        else:
            self.positional_embedding = None
        self.dropout = nn.Dropout(self._dropout)
        self._positional_encoding_type = positional_encoding_type

    def _calc_embeddings(self, x: torch.tensor):
        device = x.device
        b, t = x.size()

        tok_emb = self.embedding.calc_embeddings(x)  # token embeddings of shape (b, t, d_model)

        if self._positional_encoding_type == PositionalEncodingType.LEARNED:
            assert (
                t <= self._block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self._block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.positional_embedding(pos)  # (t, d_model)
            x = tok_emb + pos_emb
        elif self._positional_encoding_type == PositionalEncodingType.SINUSOIDAL:
            pos = torch.arange(0, t, device=device, dtype=tok_emb.dtype)  # shape (t)
            pos[::2] = torch.sin(pos[::2] / 1e4**(2*pos[::2]/self._d_model))
            pos[1::2] = torch.cos(pos[1::2] / 1e4 ** (2 * pos[1::2] / self._d_model))
            x = tok_emb + pos.reshape(-1, 1)
        else:
            raise ValueError(f'{self._positional_encoding_type} not supported')
        x = self.dropout(x)
        return x
