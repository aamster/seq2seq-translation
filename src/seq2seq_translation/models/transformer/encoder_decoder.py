import os
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from seq2seq_translation.models.transformer.decoder import DecoderTransformer
from seq2seq_translation.models.transformer.encoder import EncoderTransformer
from seq2seq_translation.models.transformer.mlp import ActivationFunction
from seq2seq_translation.models.transformer.positional_embedding import PositionalEmbeddingType


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        block_size: int,
        sos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        norm_first: bool = False,
        mlp_activation: ActivationFunction = ActivationFunction.RELU,
        positional_embedding_type: PositionalEmbeddingType = PositionalEmbeddingType.LEARNED
    ):
        super().__init__()

        self.encoder = EncoderTransformer(
            n_layers=n_layers,
            d_model=d_model,
            n_attention_heads=n_attention_heads,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            feedforward_hidden_dim=feedforward_hidden_dim,
            norm_first=norm_first,
            mlp_activation=mlp_activation,
            positional_embedding_type=positional_embedding_type
        )
        self.decoder = DecoderTransformer(
            n_layers=n_layers,
            d_model=d_model,
            n_attention_heads=n_attention_heads,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            use_cross_attention=True,
            feedforward_hidden_dim=feedforward_hidden_dim,
            norm_first=norm_first,
            mlp_activation=mlp_activation,
            positional_embedding_type=positional_embedding_type
        )
        self._block_size = block_size
        self._sos_token_id = sos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

    def forward(self, x: torch.tensor, targets: torch.tensor):
        src_key_padding_mask = (x != self._pad_token_id).bool()
        tgt_key_padding_mask = (targets != self._pad_token_id).bool()

        encoder_out = self.encoder(x=x, src_key_padding_mask=src_key_padding_mask)

        # Shift targets to the right by 1 position (teacher forcing)
        batch_size = x.shape[0]
        sos_token = torch.empty(
            batch_size, 1, dtype=torch.long, device=torch.device(os.environ["DEVICE"])
        ).fill_(self._sos_token_id)
        targets = torch.cat([sos_token, targets[:, :-1]], dim=1)

        logits = self.decoder(
            x=targets,
            memory=encoder_out,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    def generate(
        self,
        x: torch.tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        src_key_padding_mask = (x != self._pad_token_id).bool()

        encoder_out = self.encoder(x=x, src_key_padding_mask=src_key_padding_mask)
        batch_size = x.shape[0]
        generated_tokens = torch.full(
            (batch_size, 1), self._sos_token_id, dtype=torch.long
        ).to(encoder_out.device)

        input_len = x.shape[1]
        if max_new_tokens is None:
            max_new_tokens = input_len + 50 # from "Attention is all you need"

        all_logits = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if generated_tokens.size(1) <= self._block_size:
                generated_tokens_cropped = generated_tokens
            else:
                generated_tokens_cropped = generated_tokens[:, -self._block_size :]

            # forward the model to get the logits for the index in the sequence
            logits = self.decoder(
                x=generated_tokens_cropped,
                memory=encoder_out,
                memory_key_padding_mask=src_key_padding_mask,
            )
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            all_logits.append(logits)
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # Stop if all sequences in the batch generated <eos>
            if (next_token == self._eos_token_id).all():
                break
        logits = torch.cat(all_logits, dim=1)
        return generated_tokens, logits

    @property
    def num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.positional_encoding.weight.numel()
            n_params -= self.decoder.positional_encoding.weight.numel()
        return n_params
