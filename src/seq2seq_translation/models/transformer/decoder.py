from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm

from seq2seq_translation.config.transformer_config import TransformerConfig
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
            x = x + self.masked_multi_head_self_attention(
                self.layer_norm[0](x), key_padding_mask=tgt_key_padding_mask, is_causal=True
            )
            if self._use_cross_attention:
                x = x + self.multi_head_cross_attention(
                    query=self.layer_norm[1](x),
                    key=memory,
                    key_padding_mask=memory_key_padding_mask,
                )
            x = x + self.mlp(self.layer_norm[2 if self._use_cross_attention else 1](x))
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
        pad_token_idx: int,
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
            positional_encoding_type=positional_encoding_type,
            pad_token_idx=pad_token_idx
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

    @torch.no_grad()
    def generate(
        self,
        x: torch.tensor,
        eot_token_id: int,
        pad_token_id: int,
        include_input: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        return_logits: bool = False
    ):
        if include_input:
            generated_tokens = torch.tensor(x, dtype=torch.long).to(x.device)
        else:
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
        if return_logits:
            return generated_tokens, torch.cat(all_logits, dim=1)
        else:
            return generated_tokens

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

    @classmethod
    def from_pretrained(
        cls,
        config: TransformerConfig,
        model_type: str,
        vocab_size: int,
        pad_token_idx: int,
        override_args: Optional[dict] = None,
    ):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # create a from-scratch initialized minGPT model
        model = DecoderTransformer(
            n_attention_heads=config.n_head,
            n_layers=config.num_layers,
            vocab_size=vocab_size,
            d_model=config.d_model,
            block_size=config.fixed_length,
            feedforward_hidden_dim=config.feedforward_hidden_dim,
            norm_first=config.norm_first,
            mlp_activation=config.activation,
            use_cross_attention=False,
            positional_encoding_type=config.positional_encoding_type,
            pad_token_idx=pad_token_idx
        )
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        keys_map = create_pretrained_to_new_mapping(pretrained_keys=sd_keys_hf)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[keys_map[k]].shape
                with torch.no_grad():
                    sd[keys_map[k]].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[keys_map[k]].shape
                # vanilla copy over the other parameters
                with torch.no_grad():
                    sd[keys_map[k]].copy_(sd_hf[k])

        return model


def map_key(pretrained_key: str) -> str:
    """
    Map a single key from the pretrained state_dict to the corresponding key
    in your model's state_dict.
    """
    # Map token and positional embeddings.
    if pretrained_key == "transformer.wte.weight":
        return "embedding.weight"
    if pretrained_key == "transformer.wpe.weight":
        return "positional_embedding.weight"

    # Map the transformer block parameters.
    if pretrained_key.startswith("transformer.h."):
        # Split the key into parts.
        # For example, "transformer.h.0.attn.c_attn.weight" becomes:
        # ['transformer', 'h', '0', 'attn', 'c_attn', 'weight']
        parts = pretrained_key.split('.')
        layer = parts[2]  # the block number as a string
        module = parts[3]  # e.g. "ln_1", "attn", "ln_2", "mlp"

        if module == "ln_1":
            # maps to blocks.{layer}.layer_norm.0.weight/bias
            return f"blocks.{layer}.layer_norm.0.{parts[4]}"

        if module == "attn":
            # The next part distinguishes between the two attn submodules.
            sub_module = parts[4]  # either "c_attn" or "c_proj"
            if sub_module == "c_attn":
                # maps to qkv_proj.
                return f"blocks.{layer}.masked_multi_head_self_attention.qkv_proj.{parts[5]}"
            elif sub_module == "c_proj":
                # maps to output_proj.
                return f"blocks.{layer}.masked_multi_head_self_attention.output_proj.{parts[5]}"
            else:
                raise ValueError(f"Unknown attn sub-module in key: {pretrained_key}")

        if module == "ln_2":
            # maps to blocks.{layer}.layer_norm.1.weight/bias
            return f"blocks.{layer}.layer_norm.1.{parts[4]}"

        if module == "mlp":
            # parts[4] is either "c_fc" or "c_proj"
            sub_module = parts[4]
            return f"blocks.{layer}.mlp.{sub_module}.{parts[5]}"

        raise ValueError(f"Unknown module in key: {pretrained_key}")

    # Map the final layer norm.
    if pretrained_key.startswith("transformer.ln_f"):
        # For key "transformer.ln_f.weight" or "transformer.ln_f.bias"
        return f"layer_norm.{pretrained_key.split('.')[-1]}"

    # Map the lm_head.
    if pretrained_key == "lm_head.weight":
        return "lm_head.weight"

    raise ValueError(f"Unexpected key: {pretrained_key}")


def create_pretrained_to_new_mapping(pretrained_keys: list) -> dict:
    """
    Given a list of pretrained keys, return a dictionary mapping each
    pretrained key to its corresponding new key.
    """
    mapping = {}
    for key in pretrained_keys:
        new_key = map_key(key)
        mapping[key] = new_key
    return mapping
