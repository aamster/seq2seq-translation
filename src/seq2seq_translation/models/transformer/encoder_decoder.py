import os
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F, Transformer

from seq2seq_translation.models.transformer.decoder import DecoderTransformer
from seq2seq_translation.models.transformer.encoder import EncoderTransformer


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
        )
        self._block_size = block_size
        self._sos_token_id = sos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

    def forward(self, x: torch.tensor, targets: torch.tensor):
        input_pad_mask = (x != self._pad_token_id).bool()
        output_pad_mask = (targets != self._pad_token_id).bool()

        encoder_out = self.encoder(x=x, pad_mask=input_pad_mask)

        # Shift targets to the right by 1 position (teacher forcing)
        batch_size = x.shape[0]
        sos_token = torch.empty(
            batch_size, 1, dtype=torch.long, device=torch.device(os.environ["DEVICE"])
        ).fill_(self._sos_token_id)
        targets = torch.cat([sos_token, targets[:, :-1]], dim=1)

        logits = self.decoder(
            x=targets,
            encoder_out=encoder_out,
            input_pad_mask=input_pad_mask,
            output_pad_mask=output_pad_mask,
        )
        return logits

    def generate(
        self,
        x: torch.tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None
    ):
        input_pad_mask = (x != self._pad_token_id).bool()

        encoder_out = self.encoder(x=x, pad_mask=input_pad_mask)
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
                encoder_out=encoder_out,
                input_pad_mask=input_pad_mask,
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


class EncoderDecoderTransformer2(nn.Module):
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
            activation=F.gelu,
            norm_first: bool = True,
            bias: bool = False
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            nhead=n_attention_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
            bias=True   # https://github.com/pytorch/pytorch/issues/143293
        )
        self._vocab_size = vocab_size
        self._block_size = block_size
        self._sos_token_id = sos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

        self._dropout = dropout
        self._src_embedding = nn.Embedding(self._vocab_size, self.transformer.d_model)
        self._src_positional_encoding = nn.Embedding(self._block_size, self.transformer.d_model)
        self._src_embedding_dropout = nn.Dropout(self._dropout)

        self._target_embedding = nn.Embedding(self._vocab_size, self.transformer.d_model)
        self._target_positional_encoding = nn.Embedding(self._block_size, self.transformer.d_model)
        self._target_embedding_dropout = nn.Dropout(self._dropout)

        self.lm_head = nn.Linear(self.transformer.d_model, self._vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self._target_embedding.weight = self.lm_head.weight

    def forward(self, x: torch.tensor, targets: torch.tensor):
        src = self._calc_src_embedding(x=x)

        # shift target tokens 1 to the right (teacher forcing)
        sos_token = torch.empty(
            x.shape[0], 1, dtype=torch.long, device=x.device).fill_(self._sos_token_id)
        targets = torch.cat([sos_token, targets[:, :-1]], dim=1)

        target = self._calc_target_embedding(x=targets)

        target_causal_mask = torch.tril(torch.ones(target.shape[1], target.shape[1], device=target.device))
        target_causal_mask[target_causal_mask == 0] = -float('inf')
        target_causal_mask[target_causal_mask == 1] = 0

        input_pad_mask = (x == self._pad_token_id).float()
        input_pad_mask = torch.masked_fill(input_pad_mask, input_pad_mask == 1, -float('inf'))

        output_pad_mask = (targets == self._pad_token_id).float()
        output_pad_mask = torch.masked_fill(output_pad_mask, output_pad_mask == 1, -float('inf'))

        decoder_out = self.transformer(
            src=src,
            tgt=target,
            tgt_mask=target_causal_mask,
            tgt_is_causal=True,
            src_key_padding_mask=input_pad_mask,
            tgt_key_padding_mask=output_pad_mask
        )
        logits = self.lm_head(decoder_out)
        return logits

    def _calc_src_embedding(self, x: torch.tensor):
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_embedding = self._src_embedding(x) # token embeddings of shape (b, t, d_model)
        pos_emb = self._src_positional_encoding(pos)  # (t, d_model)
        x = self._src_embedding_dropout(tok_embedding + pos_emb)
        return x

    def _calc_target_embedding(self, x: torch.tensor):
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_embedding = self._target_embedding(x) # token embeddings of shape (b, t, d_model)
        pos_emb = self._target_positional_encoding(pos)  # (t, d_model)
        x = self._target_embedding_dropout(tok_embedding + pos_emb)
        return x

    @torch.no_grad()
    def generate(
        self,
        x: torch.tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None
    ):
        input_pad_mask = (x == self._pad_token_id).float()
        input_pad_mask = torch.masked_fill(input_pad_mask, input_pad_mask == 1, -float('inf'))

        src = self._calc_src_embedding(x=x)
        encoder_out = self.transformer.encoder(src=src, src_key_padding_mask=input_pad_mask)

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

            generated_tokens_cropped = self._calc_target_embedding(x=generated_tokens_cropped)

            # forward the model to get the logits for the index in the sequence
            tgt_mask = torch.tril(
                torch.ones(generated_tokens_cropped.shape[1], generated_tokens_cropped.shape[1], device=generated_tokens_cropped.device))
            tgt_mask[tgt_mask == 0] = -float('inf')
            tgt_mask[tgt_mask == 1] = 0
            decoder_out = self.transformer.decoder(
                tgt=generated_tokens_cropped,
                memory=encoder_out,
                memory_key_padding_mask=input_pad_mask,
                tgt_mask=tgt_mask,
                tgt_is_causal=True
            )
            logits = self.lm_head(decoder_out)

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
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self._src_positional_encoding.weight.numel()
            n_params -= self._target_positional_encoding.weight.numel()
        return n_params
