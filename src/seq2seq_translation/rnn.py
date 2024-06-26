from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from seq2seq_translation.attention import BahdanauAttention, AttentionType, \
    CosineSimilarityAttention


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
        dropout: float = 0.0
    ):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
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
        num_embeddings: int,
        pad_idx: Optional[int] = None,
        use_context_vector: bool = True,
        freeze_embedding_layer: bool = False,
        embedding_dim: int = 128,
        context_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,

    ):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        if use_context_vector:
            gru_input_size = embedding_dim + context_size
        else:
            gru_input_size = embedding_dim

        self.gru = nn.GRU(gru_input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.Wh = nn.Linear(encoder_output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self._use_context_vector = use_context_vector
        self._max_len = max_len
        self._sos_token_id = sos_token_id
        self._C = output_size

    def forward(self, encoder_hidden, encoder_outputs=None, target_tensor=None):
        decoder_input, decoder_hidden, decoder_outputs = self._initialize_forward(
            encoder_hidden=encoder_hidden
        )

        T = target_tensor.shape[1] if target_tensor is not None else self._max_len
        for t in range(T):
            decoder_output, decoder_hidden = self.forward_step(
                input=decoder_input,
                hidden=decoder_hidden,
                context=self._get_context(hidden=encoder_hidden)
            )
            decoder_outputs.append(decoder_output)
            decoder_input = self._get_y_t(
                decoder_output=decoder_output,
                target_tensor=target_tensor,
                t=t
            )

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden

    @staticmethod
    def _get_context(hidden, **kwargs):
        return hidden.permute(1, 0, 2)

    @staticmethod
    def _get_y_t(
        decoder_output: torch.Tensor,
        t: int,
        target_tensor: Optional[torch.Tensor] = None
    ):
        if target_tensor is not None:
            # Teacher forcing: Feed the target as the next input
            y_t = target_tensor[:, t].unsqueeze(1)  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            _, topi = decoder_output.topk(1)
            y_t = topi.squeeze(-1).detach()  # detach from history as input
        return y_t

    def _initialize_forward(self, encoder_hidden: torch.Tensor):
        batch_size = encoder_hidden.shape[1]

        decoder_hidden = encoder_hidden.transpose(0, 1).reshape(self.gru.num_layers, batch_size, -1)
        decoder_hidden = self.Wh(decoder_hidden)
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        max_len: int,
        attention_type: AttentionType,
        encoder_output_size: int,
        num_embeddings: int,
        sos_token_id: int,
        pad_idx: Optional[int] = None,
        attention_size: int = 256,
        encoder_bidirectional: bool = False,
        freeze_embedding_layer: bool = False,
        embedding_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
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
            num_layers=num_layers
        )
        if attention_type == AttentionType.BahdanauAttention:
            self.attention = BahdanauAttention(
                hidden_size=hidden_size,
                encoder_bidirectional=encoder_bidirectional
            )
        elif attention_type == AttentionType.CosineSimilarityAttention:
            self.attention = CosineSimilarityAttention(
                encoder_output_size=encoder_output_size,
                query_dim=hidden_size,
                Dv=attention_size,
                dropout=dropout
            )
        else:
            raise ValueError(f'Unknown attention type {attention_type}')

    def forward(self, encoder_hidden: torch.tensor, encoder_outputs=None, target_tensor=None,
                return_attentions: bool = False):
        if encoder_outputs is None:
            raise ValueError(f'encoder outputs must be given for {type(self)}')

        decoder_input, decoder_hidden, _ = self._initialize_forward(
            encoder_hidden=encoder_hidden
        )
        attentions = []

        batch_size = decoder_input.shape[0]
        T = target_tensor.shape[1] if target_tensor is not None else self._max_len
        decoder_outputs = torch.zeros(size=(batch_size, T, self._C),
                                      device=encoder_hidden.device)

        for t in range(T):
            attention_weights = self.attention(
                query=decoder_hidden,
                x=encoder_outputs
            )
            if return_attentions:
                attentions.append(attention_weights)

            if isinstance(self.attention, BahdanauAttention):
                context = self._get_context(
                    attn_weights=attention_weights,
                    encoder_outputs=encoder_outputs
                )
            elif isinstance(self.attention, CosineSimilarityAttention):
                context = attention_weights
            else:
                raise ValueError(f'Unknown attention {type(self.attention)}')
            decoder_output, decoder_hidden = self.forward_step(
                input=decoder_input,
                hidden=decoder_hidden,
                context=context
            )
            decoder_input = self._get_y_t(
                decoder_output=decoder_output,
                target_tensor=target_tensor,
                t=t
            )
            decoder_outputs[:, t] = decoder_output.squeeze(1)

        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        if return_attentions:
            attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def _get_context(self, **kwargs):
        attn_weights = kwargs.get('attn_weights')
        if attn_weights is None:
            raise ValueError('Attention weights need to be provided')

        encoder_outputs = kwargs.get('encoder_outputs')
        if encoder_outputs is None:
            raise ValueError('encoder_outputs should not be None')

        keys = encoder_outputs
        context = torch.bmm(attn_weights, keys)
        return context
