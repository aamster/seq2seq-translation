import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        qkv_dim: int,
        n_head: int,
        is_causal: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * qkv_dim)
        # output projection
        self.c_proj = nn.Linear(qkv_dim, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.is_causal = is_causal
        self.qkv_dim = qkv_dim

    def forward(self, **kwargs):
        raise NotImplementedError

    def _calc_attention(
        self,
        q: torch.tensor,
        k: torch.tensor,
        v: torch.tensor,
        B: int,
        T_q: int,
        T_k: int,
        query_pad_mask: Optional[torch.tensor] = None,
        key_pad_mask: Optional[torch.tensor] = None,
    ):
        # Reshape and split into heads
        q = q.view(B, T_q, self.n_head, self.qkv_dim // self.n_head).transpose(1, 2)
        k = k.view(B, T_k, self.n_head, self.qkv_dim // self.n_head).transpose(1, 2)
        v = v.view(B, T_k, self.n_head, self.qkv_dim // self.n_head).transpose(1, 2)

        if key_pad_mask is not None:
            key_pad_mask = key_pad_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T_k, 1)
            k = k.masked_fill(~key_pad_mask, 0)
            v = v.masked_fill(~key_pad_mask, 0)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=self.is_causal,
            dropout_p=self.dropout if self.training else 0,
        )

        if query_pad_mask is not None:
            query_pad_mask = query_pad_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T_q, 1)
            y = y.masked_fill(~query_pad_mask, 0)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T_q, self.qkv_dim)

        # output projection
        y = self.proj_dropout(self.c_proj(y))

        return y


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(
        self,
        n_embd: int,
        qkv_dim: int,
        n_head: int,
        is_causal: bool,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_head=n_head,
            qkv_dim=qkv_dim,
            n_embd=n_embd,
            is_causal=is_causal,
            dropout=dropout,
        )

    def forward(self, x: torch.tensor, pad_mask: torch.tensor):
        B, T, C = x.size()

        # calculates q, k, v in a single operation rather than in 3 separate operations for
        # efficiency but is equivalent
        q, k, v = self.c_attn(x).split(self.qkv_dim, dim=2)

        y = self._calc_attention(
            q=q, k=k, v=v, B=B, T_q=T, T_k=T, query_pad_mask=pad_mask, key_pad_mask=pad_mask
        )

        return y


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, n_embd: int, qkv_dim, n_head: int, dropout: float = 0.0):
        super().__init__(
            n_head=n_head,
            qkv_dim=qkv_dim,
            n_embd=n_embd,
            is_causal=False,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        key_pad_mask: torch.tensor,
        query_pad_mask: torch.tensor,
    ):
        B, T_query, C = query.size()
        _, T_key, _ = key.size()

        combined = torch.cat([query, key], dim=1)  # (B, T_query + T_key, n_embd)
        q, k, v = self.c_attn(combined).split(self.qkv_dim, dim=2)

        q = q[:, :T_query]  # (B, T_query, attention_dim)
        k = k[:, T_query:]  # (B, T_key, attention_dim)
        v = v[:, T_query:]  # (B, T_key, attention_dim)

        y = self._calc_attention(
            q=q,
            k=k,
            v=v,
            B=B,
            T_q=T_query,
            T_k=T_key,
            query_pad_mask=query_pad_mask,
            key_pad_mask=key_pad_mask,
        )
        return y


class MLP(nn.Module):

    def __init__(self, n_embd: int, hidden_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, hidden_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        qkv_dim: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.layer_norm = nn.ModuleList([LayerNorm(n_embd) for _ in range(2)])
        self.multi_head_attention = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_attention_heads,
            is_causal=False,
            dropout=dropout,
            qkv_dim=qkv_dim,
        )
        self.mlp = MLP(
            n_embd=n_embd, dropout=dropout, hidden_dim=feedforward_hidden_dim
        )

    def forward(self, x, pad_mask: torch.tensor):
        x = x + self.multi_head_attention(self.layer_norm[0](x), pad_mask=pad_mask)
        x = x + self.mlp(self.layer_norm[1](x))
        return x


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        qkv_dim: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        feedforward_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.masked_multi_head_self_attention = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_attention_heads,
            is_causal=True,
            dropout=dropout,
            qkv_dim=qkv_dim,
        )
        layer_norms = [
            LayerNorm(n_embd) for _ in range(4 if use_cross_attention else 2)
        ]
        if use_cross_attention:
            self.multi_head_cross_attention = MultiHeadCrossAttention(
                n_embd=n_embd,
                n_head=n_attention_heads,
                dropout=dropout,
                qkv_dim=qkv_dim,
            )

        self.mlp = MLP(
            n_embd=n_embd, dropout=dropout, hidden_dim=feedforward_hidden_dim
        )
        self.layer_norm = nn.ModuleList(layer_norms)
        self._use_cross_attention = use_cross_attention

    def forward(
        self,
        x: torch.tensor,
        query_pad_mask: torch.tensor,
        key_pad_mask: torch.tensor,
        encoder_out: Optional[torch.tensor] = None,
    ):
        if self._use_cross_attention:
            if encoder_out is None:
                raise ValueError("must provide encoder_out to use cross attention")
        x = x + self.masked_multi_head_self_attention(
            self.layer_norm[0](x), pad_mask=query_pad_mask
        )
        if self._use_cross_attention:
            x = x + self.multi_head_cross_attention(
                query=self.layer_norm[1](x),
                key=self.layer_norm[2](encoder_out),
                query_pad_mask=query_pad_mask,
                key_pad_mask=key_pad_mask,
            )
        x = x + self.mlp(self.layer_norm[3 if self._use_cross_attention else 1](x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._block_size = block_size
        self._n_embd = n_embd
        self._dropout = dropout
        self._n_attention_heads = n_attention_heads
        self._n_layers = n_layers

        self.embedding = nn.Embedding(self._vocab_size, self._n_embd)
        self.positional_encoding = nn.Embedding(self._block_size, self._n_embd)
        self.dropout = nn.Dropout(self._dropout)

    def _calc_embeddings(self, x: torch.tensor):
        device = x.device
        b, t = x.size()
        assert (
            t <= self._block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self._block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.embedding(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.positional_encoding(pos)  # (t, n_embd)
        x = self.dropout(tok_emb + pos_emb)
        return x


class EncoderTransformer(Transformer):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        qkv_dim: int = 64,
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            n_embd=n_embd,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
        )
        self.multi_head_attention = nn.ModuleList(
            [
                _EncoderBlock(
                    n_embd=n_embd,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    qkv_dim=qkv_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.tensor, pad_mask: torch.tensor):
        x = self._calc_embeddings(x=x)
        for block in self.multi_head_attention:
            x = block(x, pad_mask=pad_mask)
        return x


class DecoderTransformer(Transformer):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        feedforward_hidden_dim: int = 2048,
        qkv_dim: int = 64,
    ):
        super().__init__(
            n_attention_heads=n_attention_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self._use_cross_attention = use_cross_attention
        self.blocks = nn.ModuleList(
            [
                _DecoderBlock(
                    n_embd=n_embd,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    use_cross_attention=use_cross_attention,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    qkv_dim=qkv_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(self._n_embd, self._vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.embedding.weight = self.lm_head.weight

        self.layer_norm = LayerNorm(self._n_embd)

    def forward(
        self,
        x: torch.tensor,
        query_pad_mask: torch.tensor,
        key_pad_mask: torch.tensor,
        encoder_out: Optional[torch.tensor] = None,
    ):
        if self._use_cross_attention and encoder_out is None:
            raise ValueError("must provide encoder_out if use_cross_attention")

        x = self._calc_embeddings(x=x)
        for block in self.blocks:
            x = block(
                x=x,
                encoder_out=encoder_out,
                query_pad_mask=query_pad_mask,
                key_pad_mask=key_pad_mask,
            )
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        sos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        qkv_dim: int = 64,
    ):
        super().__init__()

        self.encoder = EncoderTransformer(
            n_layers=n_layers,
            n_embd=n_embd,
            n_attention_heads=n_attention_heads,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            feedforward_hidden_dim=feedforward_hidden_dim,
            qkv_dim=qkv_dim,
        )
        self.decoder = DecoderTransformer(
            n_layers=n_layers,
            n_embd=n_embd,
            n_attention_heads=n_attention_heads,
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=dropout,
            use_cross_attention=True,
            feedforward_hidden_dim=feedforward_hidden_dim,
            qkv_dim=qkv_dim,
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
            query_pad_mask=output_pad_mask,
            key_pad_mask=input_pad_mask,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        input_pad_mask = (x != self._pad_token_id).bool()

        encoder_out = self.encoder(x=x, pad_mask=input_pad_mask)
        batch_size = x.shape[0]
        generated_tokens = torch.full(
            (batch_size, 1), self._sos_token_id, dtype=torch.long
        ).to(encoder_out.device)

        input_len = x.shape[1]
        max_new_tokens = input_len + 50 # from "Attention is all you need"

        all_logits = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            if generated_tokens.size(1) <= self._block_size:
                generated_tokens_cropped = generated_tokens
            else:
                generated_tokens_cropped = generated_tokens[:, -self._block_size :]

            output_pad_mask = (generated_tokens_cropped != self._pad_token_id).bool()

            # forward the model to get the logits for the index in the sequence
            logits = self.decoder(
                x=generated_tokens_cropped,
                encoder_out=encoder_out,
                query_pad_mask=output_pad_mask,
                key_pad_mask=input_pad_mask,
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