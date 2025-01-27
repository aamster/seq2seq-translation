from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm

from seq2seq_translation.models.transformer.multi_head_attention import MultiHeadSelfAttention, \
    MultiHeadCrossAttention
from seq2seq_translation.models.transformer._transformer import _Transformer
from seq2seq_translation.models.transformer.mlp import MLP, ActivationFunction
import torch.nn.functional as F

from seq2seq_translation.models.transformer.positional_encoding import PositionalEncodingType


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
            x = self.layer_norm[2 if self._use_cross_attention else 1](x)
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
        mlp_activation: ActivationFunction = ActivationFunction.RELU,
        positional_encoding_type: PositionalEncodingType = PositionalEncodingType.LEARNED
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
            positional_encoding_type=positional_encoding_type
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

    def generate(
        self,
        x: torch.tensor,
        eot_token_id: int,
        pad_token_id: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        generated_tokens = torch.tensor([], dtype=torch.long).to(x.device)

        context = x

        input_len = x.shape[1]
        if max_new_tokens is None:
            max_new_tokens = input_len + 50 # from "Attention is all you need"

        all_logits = []
        for i in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            key_padding_mask = (context != pad_token_id).bool()
            logits = self(
                x=context,
                tgt_key_padding_mask=key_padding_mask
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
            context = torch.cat([context, next_token], dim=1)
            # Stop if all sequences in the batch generated <eos>
            if (next_token == eot_token_id).all():
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
            if self.positional_embedding is not None:
                n_params -= self.positional_embedding.weight.numel()
        return n_params