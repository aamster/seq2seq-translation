from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Model

from seq2seq_translation.attention import BahdanauAttention, AttentionType, \
    CosineSimilarityAttention


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        embedding_dim=128,
        dropout_p=0.1,
        bidirectional: bool = False,
        embedding_model: Optional[T5Model] = None,
        freeze_embedding_layer: bool = False
    ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if embedding_model is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=embedding_model.encoder.embed_tokens.weight,
                freeze=freeze_embedding_layer
            )
            embedding_dim = self.embedding.weight.shape[1]
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        self._embedding_model = embedding_model

    def forward(self, input):
        embedded = self.embedding(input)
        if self._embedding_model is None:
            embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        max_len: int,
        encoder_hidden_size: int,
        use_context_vector: bool = True,
        dropout_p=0.1,
        embedding_model: Optional[T5Model] = None,
        freeze_embedding_layer: bool = False,
        embedding_dim: int = 128,
        context_size: int = 128
    ):
        super(DecoderRNN, self).__init__()
        if embedding_model is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=embedding_model.encoder.embed_tokens.weight,
                freeze=freeze_embedding_layer
            )
            embedding_dim = self.embedding.weight.shape[1]
        else:
            self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_dim)

        if use_context_vector:
            gru_input_size = embedding_dim + context_size
        else:
            gru_input_size = embedding_dim

        self.gru = nn.GRU(gru_input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.Wh = nn.Linear(encoder_hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self._use_context_vector = use_context_vector
        self._max_len = max_len
        self._embedding_model = embedding_model

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

        decoder_hidden = encoder_hidden.reshape(1, batch_size, -1)
        decoder_hidden = self.Wh(decoder_hidden)
        decoder_input = torch.empty(
            batch_size, 1,
            dtype=torch.long,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ).fill_(0)
        return decoder_input, decoder_hidden, []

    def forward_step(self, input, hidden, context):
        """

        :param input: The previous word or predicted word (shape B x 1)
        :param hidden: Decoder hidden state (h_{t-1}). (shape D x B x H)
        :param context: context vector
        :return:
        """
        embedded = self.embedding(input)
        if self._embedding_model is None:
            embedded = self.dropout(embedded)

        if self._use_context_vector:
            if context.shape[1] != 1:
                context = context.reshape(context.shape[0], 1, -1)
            gru_input = torch.cat((embedded, context), dim=2)
        else:
            gru_input = embedded

        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output)
        return output, hidden


class AttnDecoderRNN(DecoderRNN):
    def __init__(
        self,
        hidden_size,
        output_size,
        max_len: int,
        attention_type: AttentionType,
        encoder_output_size: int,
        attention_size: int = 256,
        dropout_p=0.1,
        encoder_bidirectional: bool = False,
        embedding_model: Optional[T5Model] = None,
        freeze_embedding_layer: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            output_size=output_size,
            max_len=max_len,
            dropout_p=dropout_p,
            embedding_model=embedding_model,
            freeze_embedding_layer=freeze_embedding_layer,
            context_size=attention_size,
            encoder_hidden_size=(
                2*encoder_output_size if encoder_bidirectional else encoder_output_size)
        )
        if attention_type == AttentionType.BahdanauAttention:
            self.attention = BahdanauAttention(
                hidden_size=hidden_size,
                encoder_bidirectional=encoder_bidirectional
            )
        elif attention_type == AttentionType.CosineSimilarityAttention:
            self.attention = CosineSimilarityAttention(
                encoder_output_size=(
                    2 * encoder_output_size if encoder_bidirectional else encoder_output_size),
                decoder_hidden_size=hidden_size,
                Dv=attention_size
            )
        else:
            raise ValueError(f'Unknown attention type {attention_type}')

    def forward(self, encoder_hidden, encoder_outputs=None, target_tensor=None):
        if encoder_outputs is None:
            raise ValueError(f'encoder outputs must be given for {type(self)}')

        decoder_input, decoder_hidden, decoder_outputs = self._initialize_forward(
            encoder_hidden=encoder_hidden
        )
        attentions = []

        T = target_tensor.shape[1] if target_tensor is not None else self._max_len
        for t in range(T):
            attention_weights = self.attention(
                query=decoder_hidden,
                x=encoder_outputs
            )
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
            decoder_outputs.append(decoder_output)
            decoder_input = self._get_y_t(
                decoder_output=decoder_output,
                target_tensor=target_tensor,
                t=t
            )

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
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
