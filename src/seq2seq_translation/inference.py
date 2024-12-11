import abc
import logging
import sys

import torch
import torch.nn.functional as F

from seq2seq_translation.models.rnn import (
    EncoderRNN,
    DecoderRNN,
    AttnDecoderRNN,
    EncoderDecoderRNN,
)
from seq2seq_translation.models.transformer.encoder_decoder import EncoderDecoderTransformer
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)


def get_logger():
    logging.basicConfig(
        format="%(message)s",
        # datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    return logger


logger = get_logger()


class SequenceGenerator(abc.ABC):
    def __init__(
        self,
        encoder_decoder: EncoderDecoderRNN | EncoderDecoderTransformer,
        tokenizer: SentencePieceTokenizer,
    ):
        self._encoder_decoder = encoder_decoder
        self._tokenizer = tokenizer

    @abc.abstractmethod
    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        raise NotImplementedError


class GreedySequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        encoder_decoder: EncoderDecoderRNN | EncoderDecoderTransformer,
        tokenizer: SentencePieceTokenizer,
        max_length: int = 72,
    ):
        super().__init__(
            encoder_decoder=encoder_decoder,
            tokenizer=tokenizer,
        )
        self._max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        decoder_output, _, _ = self._encoder_decoder(
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
        encoder_decoder: EncoderDecoderRNN | EncoderDecoderTransformer,
        tokenizer: SentencePieceTokenizer,
        beam_width=10,
        max_length=72,
    ):
        super().__init__(encoder_decoder=encoder_decoder, tokenizer=tokenizer)
        self.beam_width = beam_width
        self.max_length = max_length

    def generate(self, input_tensor: torch.tensor, input_lengths: list[int]):
        src_tensor = input_tensor.unsqueeze(0)

        encoder_outputs, encoder_hidden = self._encoder_decoder.encoder(
            src_tensor, input_lengths=input_lengths
        )

        initial_decoder_input, decoder_hidden, _ = (
            self._encoder_decoder.decoder.initialize_forward(
                encoder_hidden=encoder_hidden
            )
        )

        beams = [(initial_decoder_input, decoder_hidden, initial_decoder_input, 0)]
        all_candidates = []

        for beam_search_iter in range(self.max_length):
            new_beams = []
            if all(
                [
                    b[2][:, -1].item() == self._tokenizer.processor.eos_id()
                    for b in beams
                ]
            ):
                logger.debug("terminating")
                break
            logger.debug(f"\nbeam search iter {beam_search_iter}")
            logger.debug("=" * 11)

            for decoder_input, decoder_hidden, decoded_sequence, score in beams:
                if decoded_sequence[:, -1].item() == self._tokenizer.processor.eos_id():
                    continue
                with torch.no_grad():
                    if isinstance(self._encoder_decoder.decoder, AttnDecoderRNN):
                        topk_indices, topk_scores, _, _, new_decoder_hidden = (
                            self._encoder_decoder.decoder.decode_step(
                                decoder_input=decoder_input,
                                decoder_hidden=decoder_hidden,
                                encoder_hidden=encoder_hidden,
                                encoder_outputs=encoder_outputs,
                                k=self.beam_width,
                                softmax_scores=True,
                            )
                        )
                    elif isinstance(self._encoder_decoder.decoder, DecoderRNN):
                        topk_indices, topk_scores, _, new_decoder_hidden = (
                            self._encoder_decoder.decoder.decode_step(
                                decoder_input=decoder_input,
                                decoder_hidden=decoder_hidden,
                                encoder_hidden=encoder_hidden,
                                encoder_outputs=encoder_outputs,
                                k=self.beam_width,
                                softmax_scores=True,
                            )
                        )

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
                    if next_token_id == self._tokenizer.processor.eos_id():
                        all_candidates.append(
                            (
                                next_token_id,
                                new_decoder_hidden,
                                new_decoded_sequence,
                                new_score,
                            )
                        )
                    else:
                        new_beams.append(
                            (
                                next_token_id,
                                new_decoder_hidden,
                                new_decoded_sequence,
                                new_score,
                            )
                        )
            logger.debug("\nNew beams:")
            for new_beam in new_beams:
                logger.debug(
                    " ".join(
                        [self._tokenizer.decode(new_beam[2]), f"{new_beam[3]:.3f}"]
                    )
                )

            # Sort new beams by score and select top k
            beams = sorted(new_beams, key=lambda x: x[-1], reverse=True)[
                : self.beam_width
            ]

            logger.debug("\nBeams:")
            for beam in beams:
                _, _, decoded_sequence, score = beam
                logger.debug(
                    " ".join([self._tokenizer.decode(decoded_sequence), f"{score:.3f}"])
                )

            if all_candidates:
                logger.debug(f"\ncompleted\n\n")
                for completed in all_candidates:
                    logger.debug(
                        " ".join(
                            [
                                self._tokenizer.decode(completed[2]),
                                f"{completed[3]:.3f}",
                            ]
                        )
                    )
        # Combine the remaining beams with all_candidates
        all_candidates.extend(beams)

        # Sort candidates by score and return the best sequences
        preds = sorted(all_candidates, key=lambda x: x[-1], reverse=True)
        preds = [[self._tokenizer.decode(x[2]), x[3]] for x in preds]

        logger.debug("\nfinal list of preds\n===========\n")
        for pred in preds:
            logger.debug(" ".join([pred[0], f"{pred[1]:.3f}"]))

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
