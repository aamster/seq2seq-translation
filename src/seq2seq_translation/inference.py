import abc
import os
import sys

import torch
import torch.nn.functional as F

from seq2seq_translation.models.rnn import (
    EncoderRNN,
    DecoderRNN,
    AttnDecoderRNN,
    EncoderDecoderRNN,
)
from seq2seq_translation.models.transformer.decoder import DecoderTransformer, generate
from seq2seq_translation.models.transformer.encoder_decoder import EncoderDecoderTransformer
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)
from loguru import logger

from seq2seq_translation.utils.model_util import model_isinstance

logger.remove()

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
logger.add(sys.stderr, level=LOG_LEVEL)


class SequenceGenerator(abc.ABC):
    def __init__(
        self,
        model: EncoderDecoderRNN | EncoderDecoderTransformer | DecoderTransformer,
        tokenizer: SentencePieceTokenizer,
    ):
        self._model = model
        self._tokenizer = tokenizer

    @abc.abstractmethod
    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        raise NotImplementedError


class GreedySequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        model: EncoderDecoderRNN | EncoderDecoderTransformer,
        tokenizer: SentencePieceTokenizer,
        max_length: int = 72,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
        )
        self._max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        decoder_output, _, _ = self._model(
            input_tensor.unsqueeze(0), input_lengths=input_lengths
        )

        _, topi = decoder_output.topk(k=1, dim=-1)
        tokens = topi.squeeze(-1)
        pred = self._tokenizer.decode(tokens)

        batch_size = input_tensor.shape[0]

        if batch_size == 1:
            pred = [pred]
        return pred


class BeamSearchSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        model: EncoderDecoderRNN | EncoderDecoderTransformer,
        tokenizer: SentencePieceTokenizer,
        beam_width=4,
        max_length=72,
    ):
        super().__init__(model=model, tokenizer=tokenizer)
        self.beam_width = beam_width
        self.max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        if LOG_LEVEL == 'TRACE':
            logger.trace(f'input: {self._tokenizer.decode(input_tensor)}')
        src_tensor = input_tensor.unsqueeze(0)

        if model_isinstance(self._model, (EncoderDecoderRNN, EncoderDecoderTransformer)):
            encoder_outputs, encoder_hidden = self._model.encoder(
                src_tensor, input_lengths=input_lengths
            )

            initial_decoder_input, decoder_hidden, _ = (
                self._model.decoder.initialize_forward(
                    encoder_hidden=encoder_hidden
                )
            )
            beams = [{
                'decoder_input': initial_decoder_input,
                'decoder_hidden': decoder_hidden,
                'decoded_sequence': initial_decoder_input,
                'score': 0
            }]
        else:
            encoder_outputs = None
            encoder_hidden = None
            beams = [{
                'decoder_input': src_tensor,
                'decoder_hidden': None,
                'decoded_sequence': torch.tensor([], device=input_tensor.device, dtype=torch.long),
                'score': 0
            }]

        all_candidates = []

        for beam_search_iter in range(self.max_length):
            new_beams = []
            if all(
                [
                    len(b['decoded_sequence']) > 0 and b['decoded_sequence'][:, -1].item() == self._tokenizer.processor.eos_id()
                    for b in beams
                ]
            ):
                logger.trace("terminating")
                break
            logger.trace(f"\nbeam search iter {beam_search_iter}")
            logger.trace("=" * 11)

            for beam in beams:
                decoder_input = beam['decoder_input']
                decoder_hidden = beam['decoder_hidden']
                decoded_sequence = beam['decoded_sequence']
                score = beam['score']

                if len(decoded_sequence) > 0 and decoded_sequence[:, -1].item() == self._tokenizer.processor.eos_id():
                    continue
                with torch.no_grad():
                    if model_isinstance(self._model, EncoderDecoderRNN):
                        if model_isinstance(self._model.decoder, AttnDecoderRNN):
                            topk_indices, topk_scores, _, _, new_decoder_hidden = (
                                self._model.decoder.decode_step(
                                    decoder_input=decoder_input,
                                    decoder_hidden=decoder_hidden,
                                    encoder_hidden=encoder_hidden,
                                    encoder_outputs=encoder_outputs,
                                    k=self.beam_width,
                                    softmax_scores=True,
                                )
                            )
                        elif model_isinstance(self._model.decoder, DecoderRNN):
                            topk_indices, topk_scores, _, new_decoder_hidden = (
                                self._model.decoder.decode_step(
                                    decoder_input=decoder_input,
                                    decoder_hidden=decoder_hidden,
                                    encoder_hidden=encoder_hidden,
                                    encoder_outputs=encoder_outputs,
                                    k=self.beam_width,
                                    softmax_scores=True,
                                )
                            )
                    elif model_isinstance(self._model, DecoderTransformer):
                        _, logits  = generate(
                            model=self._model,
                            x=decoder_input,
                            eot_token_id=self._tokenizer.eot_idx,
                            pad_token_id=self._tokenizer.pad_idx,
                            top_k=self.beam_width,
                            max_new_tokens=1,
                            return_logits=True
                        )
                        probs = F.softmax(logits, dim=-1)
                        topk_indices = torch.nonzero(probs)[:, 1].unsqueeze(0).unsqueeze(0)
                        topk_scores = probs[0][topk_indices[0, 0]].unsqueeze(0).unsqueeze(0)
                        new_decoder_hidden = None
                if torch.any(
                    topk_indices[:, :, : self.beam_width]
                    == self._tokenizer.processor.eos_id()
                ):
                    new_beam_indices = torch.where(
                        topk_indices[:, :, : self.beam_width]
                        == self._tokenizer.processor.eos_id()
                    )[1]
                else:
                    new_beam_indices = range(self.beam_width)

                # Create new beams by appending each of the top k tokens
                for k in new_beam_indices:
                    next_token_id = topk_indices[:, :, k]
                    new_decoded_sequence = torch.cat(
                        [decoded_sequence, next_token_id], dim=1
                    )
                    new_score = score + torch.log(topk_scores[:, :, k]).item()
                    if model_isinstance(self._model, EncoderDecoderRNN):
                        new_decoder_input = next_token_id
                    elif model_isinstance(self._model, DecoderTransformer):
                        new_decoder_input = torch.cat((decoder_input, next_token_id), dim=1)
                    else:
                        raise NotImplementedError
                    if next_token_id == self._tokenizer.processor.eos_id():
                        all_candidates.append(
                            {
                                'decoder_input': new_decoder_input,
                                'decoder_hidden': new_decoder_hidden,
                                'decoded_sequence': new_decoded_sequence,
                                'score': new_score,
                            }
                        )
                    else:
                        new_beams.append(
                            {
                                'decoder_input': new_decoder_input,
                                'decoder_hidden': new_decoder_hidden,
                                'decoded_sequence': new_decoded_sequence,
                                'score': new_score,
                            }
                        )
            logger.trace("\nNew beams:")
            for new_beam in new_beams:
                if LOG_LEVEL == 'TRACE':
                    logger.trace(
                        " ".join(
                            [self._tokenizer.decode(new_beam['decoded_sequence']), f"{new_beam['score']:.3f}"]
                        )
                    )

            # Sort new beams by score and select top k
            beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[
                : self.beam_width
            ]

            logger.trace("\nBeams:")
            for beam in beams:
                if LOG_LEVEL == 'TRACE':
                    logger.trace(
                        " ".join([self._tokenizer.decode(beam['decoded_sequence']), f"{beam['score']:.3f}"])
                    )

            if all_candidates:
                logger.trace(f"\ncompleted\n\n")
                for completed in all_candidates:
                    if LOG_LEVEL == 'TRACE':
                        logger.trace(
                            " ".join(
                                [
                                    self._tokenizer.decode(completed['decoded_sequence']),
                                    f"{completed['score']:.3f}",
                                ]
                            )
                        )
        # Combine the remaining beams with all_candidates
        all_candidates.extend(beams)

        # Sort candidates by score and return the best sequences
        preds = sorted(all_candidates, key=lambda x: x['score'], reverse=True)
        preds = [[self._tokenizer.decode(x['decoded_sequence']), x['score']] for x in preds]

        logger.trace("\nfinal list of preds\n===========\n")
        for pred in preds:
            logger.trace(" ".join([pred[0], f"{pred[1]:.3f}"]))

        return preds


class SoftmaxWithTemperatureSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        tokenizer: SentencePieceTokenizer,
        temperature: float = 1.0,
        max_length: int = 72,
    ):
        super().__init__(encoder=encoder, decoder=decoder, tokenizer=tokenizer)
        self._temperature = temperature
        self._max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        encoder_outputs, encoder_hidden = self._encoder(
            input_tensor, input_lengths=input_lengths
        )

        decoder_input, decoder_hidden, _ = self._decoder.initialize_forward(
            encoder_hidden=encoder_hidden
        )
        decoder_outputs = []
        for _ in range(self._max_length):
            decoder_input, _, decoder_output, _, decoder_hidden = (
                self._decoder.decode_step(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_hidden=encoder_hidden,
                    encoder_outputs=encoder_outputs,
                )
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
