import abc

import torch
import torch.nn.functional as F

from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer


class SequenceGenerator(abc.ABC):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, tokenizer: SentencePieceTokenizer):
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer

    @abc.abstractmethod
    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        raise NotImplementedError


class GreedySequenceGenerator(SequenceGenerator):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, tokenizer: SentencePieceTokenizer, max_length: int = 72):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
        )
        self._max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        encoder_outputs, encoder_hidden = self._encoder(input_tensor.unsqueeze(0), input_lengths=input_lengths)

        decoder_output, _, _ = self._decoder(
            encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs,
        )

        _, topi = decoder_output.topk(k=1, dim=-1)
        tokens = topi.squeeze(-1)
        pred = self._tokenizer.decode(tokens)

        batch_size = input_tensor.shape[0]

        if batch_size == 1:
            pred = [pred]
        return pred


class BeamSearchSequenceGenerator(SequenceGenerator):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, tokenizer: SentencePieceTokenizer, beam_width=10, max_length=72):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer
        )
        self.beam_width = beam_width
        self.max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        src_tensor = input_tensor.unsqueeze(0)

        encoder_outputs, encoder_hidden = self._encoder(src_tensor, input_lengths=input_lengths)

        initial_decoder_input, decoder_hidden, _ = self._decoder.initialize_forward(
            encoder_hidden=encoder_hidden
        )

        # Initial states of the beam
        beams = [
            (initial_decoder_input, decoder_hidden, initial_decoder_input, 0)]

        for _ in range(self.max_length):
            all_candidates = []
            for decoder_input, decoder_hidden, decoded_sequence, score in beams:
                if decoder_input[:, -1].item() == self._tokenizer.processor.eos_id():
                    all_candidates.append((decoder_input, decoder_hidden, decoded_sequence, score))
                    continue

                # Predict the next token
                with torch.no_grad():
                    if isinstance(self._decoder, AttnDecoderRNN):
                        topk_indices, topk_scores, _, _, new_decoder_hidden = self._decoder.decode_step(
                            decoder_input=decoder_input,
                            decoder_hidden=decoder_hidden,
                            encoder_hidden=encoder_hidden,
                            encoder_outputs=encoder_outputs,
                            k=self.beam_width,
                            softmax_scores=True
                        )
                    elif isinstance(self._decoder, DecoderRNN):
                        topk_indices, topk_scores, _, new_decoder_hidden = self._decoder.decode_step(
                            decoder_input=decoder_input,
                            decoder_hidden=decoder_hidden,
                            encoder_hidden=encoder_hidden,
                            encoder_outputs=encoder_outputs,
                            k=self.beam_width,
                            softmax_scores=True
                        )

                # Create new beams
                for i in range(self.beam_width):
                    next_token_id = topk_indices[:, :, i]
                    new_decoded_sequence = torch.cat([decoded_sequence, next_token_id], dim=1)
                    new_score = score + torch.log(topk_scores[:, :, i]).item()
                    all_candidates.append((next_token_id, new_decoder_hidden,  new_decoded_sequence, new_score))

            # Keep only the top scoring beams
            beams = sorted(all_candidates, key=lambda x: x[-1], reverse=True)[:self.beam_width]

        # preds sorted by highest prob
        preds = sorted(beams, key=lambda x: x[-1])[::-1]

        preds = [[self._tokenizer.decode(x[2]), x[3]] for x in preds]
        return preds


class SoftmaxWithTemperatureSequenceGenerator(SequenceGenerator):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, tokenizer: SentencePieceTokenizer, temperature: float = 1.0, max_length: int = 72):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer
        )
        self._temperature = temperature
        self._max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        encoder_outputs, encoder_hidden = self._encoder(input_tensor, input_lengths=input_lengths)

        decoder_input, decoder_hidden, _ = self._decoder.initialize_forward(
            encoder_hidden=encoder_hidden
        )
        decoder_outputs = []
        for _ in range(self._max_length):
            decoder_input, _, decoder_output, _, decoder_hidden = self._decoder.decode_step(
                decoder_input=decoder_input,
                decoder_hidden=decoder_hidden,
                encoder_hidden=encoder_hidden,
                encoder_outputs=encoder_outputs,
            )
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        decoder_outputs = decoder_outputs / self._temperature

        probs = F.softmax(decoder_outputs, dim=-1)

        batch_size = probs.shape[0]
        num_tokens = probs.shape[-1]
        # need to reshape to 2d in order to use multinomial
        probs = probs.view(-1, num_tokens)

        sampled_tokens = torch.multinomial(probs, num_samples=1)

        # reshape back into original shape
        sampled_tokens = sampled_tokens.squeeze(-1).view(batch_size, self._max_length)

        decoded_text = self._tokenizer.decode(sampled_tokens)
        if batch_size == 1:
            decoded_text = [decoded_text]
        return decoded_text
