import os
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from seq2seq_translation.models.transformer.decoder import DecoderTransformer
from seq2seq_translation.models.transformer.encoder import EncoderTransformer
from seq2seq_translation.models.transformer.mlp import ActivationFunction
from seq2seq_translation.models.transformer.positional_encoding import (
    PositionalEncodingType,
)


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
        positional_encoding_type: PositionalEncodingType = PositionalEncodingType.LEARNED,
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
            positional_encoding_type=positional_encoding_type,
            pad_token_idx=pad_token_id,
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
            positional_encoding_type=positional_encoding_type,
            pad_token_idx=pad_token_id,
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
        src: torch.tensor,
        x: Optional[torch.tensor] = None,
        encoder_out: Optional[torch.tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        return_cross_attention_weights: bool = False,
    ):
        src_key_padding_mask = (src != self._pad_token_id).bool()

        if encoder_out is None:
            encoder_out = self.encoder(x=src, src_key_padding_mask=src_key_padding_mask)
        batch_size = src.shape[0]
        if x is None:
            generated_tokens = torch.full(
                (batch_size, 1), self._sos_token_id, dtype=torch.long
            ).to(encoder_out.device)
        else:
            generated_tokens = x

        input_len = src.shape[1]
        if max_new_tokens is None:
            max_new_tokens = input_len + 50  # from "Attention is all you need"

        all_logits = []
        all_cross_attention_weights = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if generated_tokens.size(1) <= self._block_size:
                generated_tokens_cropped = generated_tokens
            else:
                generated_tokens_cropped = generated_tokens[:, -self._block_size :]

            # forward the model to get the logits for the index in the sequence
            decoder_out = self.decoder(
                x=generated_tokens_cropped,
                memory=encoder_out,
                memory_key_padding_mask=src_key_padding_mask,
                return_cross_attention_weights=return_cross_attention_weights,
            )
            if return_cross_attention_weights:
                logits, attention_weights = decoder_out

                # extract just the new token cross attention
                if len(all_cross_attention_weights) > 0:
                    for layer in range(len(attention_weights)):
                        all_cross_attention_weights[layer] = torch.cat(
                            [
                                all_cross_attention_weights[layer],
                                attention_weights[layer][:, :, -1].unsqueeze(2),
                            ],
                            dim=2,
                        )
                else:
                    all_cross_attention_weights += [
                        x[:, :, -1].unsqueeze(2) for x in attention_weights
                    ]
            else:
                logits = decoder_out
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

        if return_cross_attention_weights:
            return generated_tokens, logits, all_cross_attention_weights
        else:
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
            n_params -= self.encoder.embedding.weight.numel()
            if self.encoder.positional_embedding is not None:
                n_params -= self.encoder.positional_embedding.weight.numel()
            if self.decoder.positional_embedding is not None:
                n_params -= self.decoder.positional_embedding.weight.numel()
        return n_params
