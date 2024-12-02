import abc
import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from seq2seq_translation.models.attention import (
    BahdanauAttention,
    AttentionType,
    CosineSimilarityAttention,
)


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size,
        pad_idx: Optional[int] = None,
        hidden_size=128,
        embedding_dim=128,
        bidirectional: bool = False,
        freeze_embedding_layer: bool = False,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_dim
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self._pad_idx = pad_idx

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            input=embedded,
            lengths=input_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, hidden = self.gru(packed_embedded)

        # Convert back into a tensor
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=self._pad_idx
        )

        return output, hidden

    @property
    def hidden_size(self):
        hidden_size = self.gru.hidden_size
        if self.gru.bidirectional:
            hidden_size *= 2
        hidden_size *= self.gru.num_layers
        return hidden_size

    @property
    def output_size(self):
        output_size = self.gru.hidden_size
        if self.gru.bidirectional:
            output_size *= 2
        return output_size


class DecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        max_len: int,
        encoder_output_size: int,
        sos_token_id: int,
        eos_token_id: int,
        num_embeddings: int,
        pad_idx: Optional[int] = None,
        use_context_vector: bool = True,
        freeze_embedding_layer: bool = False,
        embedding_dim: int = 128,
        context_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        encoder_bidirectional: bool = True,
    ):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

        if use_context_vector:
            gru_input_size = embedding_dim + context_size
        else:
            gru_input_size = embedding_dim

        self.gru = nn.GRU(
            gru_input_size, hidden_size, batch_first=True, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.Wh = nn.Linear(encoder_output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self._use_context_vector = use_context_vector
        self._max_len = max_len
        self._sos_token_id = sos_token_id
        self._C = output_size
        self._encoder_bidirectional = encoder_bidirectional
        self._pad_idx = pad_idx
        self._eos_token_id = eos_token_id

    def forward(
        self, encoder_hidden: torch.tensor, encoder_outputs=None, target_tensor=None
    ):
        decoder_input, decoder_hidden, decoder_outputs = self.initialize_forward(
            encoder_hidden=encoder_hidden
        )
        batch_size = decoder_input.shape[0]
        finished_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=encoder_hidden.device
        )

        outputs = []
        for t in range(self._max_len):
            if finished_mask.all():
                break  # Stop if all sequences are finished

            decoder_input, _, decoder_output, decoder_hidden = self.decode_step(
                decoder_input=decoder_input,
                decoder_hidden=decoder_hidden,
                encoder_hidden=encoder_hidden,
                target_tensor=target_tensor,
                t=t,
            )
            outputs.append(decoder_output.topk(k=1, dim=-1)[1].squeeze())

            finished_mask |= decoder_input == self._eos_token_id

            decoder_input = decoder_input.masked_fill(finished_mask, self._eos_token_id)

        decoder_outputs = torch.stack(outputs).T
        return decoder_outputs, decoder_hidden

    def decode_step(
        self,
        decoder_input,
        decoder_hidden,
        encoder_hidden,
        target_tensor: Optional[torch.tensor] = None,
        t: Optional[int] = None,
        encoder_outputs: Optional[torch.tensor] = None,
        k: int = 1,
        softmax_scores: bool = False,
    ):
        decoder_output, decoder_hidden = self.forward_step(
            input=decoder_input,
            hidden=decoder_hidden,
            context=self._get_context(hidden=encoder_hidden),
        )
        token, scores = self._get_y_t(
            decoder_output=decoder_output,
            target_tensor=target_tensor,
            t=t,
            k=k,
            softmax_scores=softmax_scores,
        )
        return token, scores, decoder_output, decoder_hidden

    def _get_context(self, hidden, **kwargs):
        # extracting the hidden state of the last layer
        hidden = hidden[-2:] if self._encoder_bidirectional else hidden[-1]
        return hidden.permute(1, 0, 2)

    @staticmethod
    def _get_y_t(
        decoder_output: torch.Tensor,
        t: Optional[int] = None,
        target_tensor: Optional[torch.Tensor] = None,
        k=1,
        softmax_scores: bool = False,
    ):
        if target_tensor is not None:
            if t is None:
                raise ValueError("must provide t if training")
            # Teacher forcing: Feed the target as the next input
            y_t = target_tensor[:, t].unsqueeze(1)  # Teacher forcing
            top_scores = None
        else:
            # Without teacher forcing: use its own predictions as the next input
            if softmax_scores:
                decoder_output = F.softmax(decoder_output, dim=-1)
            top_scores, topi = decoder_output.topk(k=k, dim=-1)
            y_t = topi.squeeze(-1)
            y_t = y_t.detach()  # detach from history as input

        return y_t, top_scores

    def initialize_forward(self, encoder_hidden: torch.Tensor):
        batch_size = encoder_hidden.shape[1]

        decoder_hidden = encoder_hidden.transpose(0, 1).reshape(
            self.gru.num_layers, batch_size, -1
        )
        decoder_hidden = self.Wh(decoder_hidden)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=torch.device(os.environ["DEVICE"])
        ).fill_(self._sos_token_id)
        return decoder_input, decoder_hidden, []

    def forward_step(self, input, hidden, context):
        """

        :param input: The previous word or predicted word (shape B x 1)
        :param hidden: Decoder hidden state (h_{t-1}). (shape D x B x H)
        :param context: context vector
        :return:
        """
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        if self._use_context_vector:
            if context.shape[1] != 1:
                context = context.reshape(context.shape[0], 1, -1)
            gru_input = torch.cat((embedded, context), dim=2)
        else:
            gru_input = embedded

        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output)
        output = self.dropout(output)
        return output, hidden


class AttnDecoderRNN(DecoderRNN):
    def __init__(
        self,
        hidden_size,
        output_size,
        attention_type: AttentionType,
        encoder_output_size: int,
        num_embeddings: int,
        sos_token_id: int,
        eos_token_id: int,
        pad_idx: Optional[int] = None,
        attention_size: int = 256,
        encoder_bidirectional: bool = False,
        freeze_embedding_layer: bool = False,
        embedding_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_len: Optional[int] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            output_size=output_size,
            max_len=max_len,
            dropout=dropout,
            freeze_embedding_layer=freeze_embedding_layer,
            context_size=attention_size,
            encoder_output_size=encoder_output_size,
            pad_idx=pad_idx,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sos_token_id=sos_token_id,
            num_layers=num_layers,
            eos_token_id=eos_token_id,
        )
        if attention_type == AttentionType.BahdanauAttention:
            self.attention = BahdanauAttention(
                hidden_size=hidden_size,
            )
        elif attention_type == AttentionType.CosineSimilarityAttention:
            self.attention = CosineSimilarityAttention(
                encoder_output_size=encoder_output_size,
                query_dim=hidden_size,
                Dv=attention_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attention type {attention_type}")

    def forward(
        self,
        encoder_hidden: torch.tensor,
        encoder_outputs=None,
        target_tensor=None,
        return_attentions: bool = False,
        softmax_output: bool = True,
    ):
        if encoder_outputs is None:
            raise ValueError(f"encoder outputs must be given for {type(self)}")

        decoder_input, decoder_hidden, _ = self.initialize_forward(
            encoder_hidden=encoder_hidden
        )
        attentions = []

        batch_size = decoder_input.shape[0]
        finished_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=decoder_input.device
        )

        outputs = []
        for t in range(self._max_len):
            if finished_mask.all():
                break  # Stop if all sequences are finished

            decoder_input, _, decoder_output, attention_weights, decoder_hidden = (
                self.decode_step(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_hidden=encoder_hidden,
                    encoder_outputs=encoder_outputs,
                    target_tensor=target_tensor,
                    t=t,
                )
            )

            if softmax_output:
                outputs.append(decoder_output)
            else:
                outputs.append(decoder_output.topk(k=1, dim=-1)[1].squeeze())

            finished_mask |= decoder_input == self._eos_token_id

            decoder_input = decoder_input.masked_fill(finished_mask, self._eos_token_id)

            if return_attentions:
                attentions.append(attention_weights)

        if softmax_output:
            # (batch_size, sequence_length, hidden_size)
            decoder_outputs = torch.cat(outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        else:
            decoder_outputs = torch.stack(outputs).T

        if return_attentions:
            attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def decode_step(
        self,
        decoder_input,
        decoder_hidden,
        encoder_hidden,
        encoder_outputs: Optional[torch.tensor] = None,
        target_tensor: Optional[torch.tensor] = None,
        t: Optional[int] = None,
        k: int = 1,
        softmax_scores: bool = False,
    ):
        if encoder_outputs is None:
            raise ValueError("encoder_outputs must be provided")

        mask = (encoder_outputs == self._pad_idx).all(dim=-1)

        if isinstance(self.attention, BahdanauAttention):
            attention_values, attention_weights = self.attention(
                s_t_minus_1=decoder_hidden, h_j=encoder_outputs, mask=mask
            )
        else:
            attention_values, attention_weights = self.attention(
                query=decoder_hidden, x=encoder_outputs, mask=mask
            )

        if isinstance(self.attention, BahdanauAttention):
            context = self._get_context(
                attn_weights=attention_values, encoder_outputs=encoder_outputs
            )
        elif isinstance(self.attention, CosineSimilarityAttention):
            context = attention_values
        else:
            raise ValueError(f"Unknown attention {type(self.attention)}")
        decoder_output, decoder_hidden = self.forward_step(
            input=decoder_input, hidden=decoder_hidden, context=context
        )
        token, scores = self._get_y_t(
            decoder_output=decoder_output,
            target_tensor=target_tensor,
            t=t,
            k=k,
            softmax_scores=softmax_scores,
        )
        return token, scores, decoder_output, attention_weights, decoder_hidden

    def _get_context(self, **kwargs):
        attn_weights = kwargs.get("attn_weights")
        if attn_weights is None:
            raise ValueError("Attention weights need to be provided")

        encoder_outputs = kwargs.get("encoder_outputs")
        if encoder_outputs is None:
            raise ValueError("encoder_outputs should not be None")

        keys = encoder_outputs
        context = torch.bmm(attn_weights, keys)
        return context


class EncoderDecoderRNN(nn.Module):
    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN | AttnDecoderRNN,
    ):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(
        self,
        x: torch.tensor,
        input_lengths,
        target_tensor: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        output, hidden = self._encoder(input=x, input_lengths=input_lengths)

        kwargs = {}
        if isinstance(self._decoder, AttnDecoderRNN) and return_attention_weights:
            kwargs["return_attentions"] = True
        decoder_res = self._decoder(
            encoder_outputs=output,
            encoder_hidden=hidden,
            target_tensor=target_tensor,
            **kwargs,
        )

        if len(decoder_res) == 3:
            decoder_outputs, decoder_hidden, decoder_attn = decoder_res
        else:
            decoder_outputs, decoder_hidden = decoder_res
            decoder_attn = None

        return decoder_outputs, decoder_hidden, decoder_attn

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
