import os
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
import torch


class SentencePieceTokenizer:
    def __init__(self, model_prefix: str, input_path: Optional[str] = None, vocab_size: Optional[int] = 13000):
        self._processor = spm.SentencePieceProcessor()
        self._options = dict(
            # input spec
            input=input_path,
            input_format="text",
            # output spec
            model_prefix=model_prefix,  # output filename prefix
            # algorithm spec
            # BPE alg
            model_type="bpe",
            vocab_size=vocab_size,
            # normalization
            normalization_rule_name="identity",  # ew, turn off normalization
            remove_extra_whitespaces=False,
            input_sentence_size=200000000,  # max number of training sentences
            max_sentence_length=4192,  # max number of bytes per sentence
            seed_sentencepiece_size=1000000,
            shuffle_input_sentence=True,
            # rare word treatment
            character_coverage=0.99995,
            byte_fallback=True,
            # merge rules
            split_digits=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            max_sentencepiece_length=16,
            add_dummy_prefix=True,
            allow_whitespace_only_pieces=True,
            # special tokens
            unk_id=0,  # the UNK token MUST exist
            bos_id=1,  # the others are optional, set to -1 to turn off
            eos_id=2,
            pad_id=3,
            # systems
            num_threads=os.cpu_count()  # use ~all system resources
        )
        self._model = f'{model_prefix}.model'

        if Path(self._model).exists():
            self._processor.load(self._model)
        else:
            if input_path is None:
                raise ValueError('Must provide train_path if training')
            if vocab_size is None:
                raise ValueError('Must provide vocab_size if training')
            self.train()

    @property
    def processor(self):
        return self._processor

    @property
    def vocab(self):
        return dict([self.processor.id_to_piece(idx), idx] for idx in range(self.processor.get_piece_size()))

    def train(self):
        spm.SentencePieceTrainer.train(**self._options)
        self._processor.load(self._model)

    def decode(self, token_ids: torch.tensor) -> List[str] | str:
        """
        Decode `token_ids` to a string or list of strings
        Each list of token_ids is truncated to the first occurance of the eos token

        :param token_ids: token ids to decode
        :return:
        """
        decoded = []
        if len(token_ids.shape) == 1:
            token_ids = token_ids.reshape(1, -1)
        for tokens in token_ids:
            eos_idx = torch.where(tokens == self.processor.eos_id())[0]
            if len(eos_idx) > 0:
                eos_idx = eos_idx[0]
                tokens = tokens[:eos_idx]
            decoded.append(self.processor.decode(tokens.tolist()))
        if len(decoded) == 1:
            decoded = decoded[0]
        return decoded