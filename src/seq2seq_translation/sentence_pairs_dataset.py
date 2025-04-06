from typing import Optional

import numpy as np
import torch
from tiktoken import Encoding
from torch.utils.data import Dataset

from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)


class SentencePairsDatasetFromPreprocessedTokens(Dataset):
    def __init__(
        self,
        idxs: np.ndarray,
        tokenized_offsets: np.ndarray,
        tokenized: np.memmap,
        eot_token_id: int,
        pad_token_id: int,
        source_language_tag_token_id: int,
        target_language_tag_token_id: int,
        combine_source_and_target: bool = False,
    ):
        """

        :param idxs: idxs to use from LanguagePairsDatasets
        :param tokenized_offsets: offsets into the tokenized memmap array
        :param tokenized: the memmap array storing all tokenized inputs in a single 1d array
        :param eot_token_id
        """

        self._idxs = idxs
        self._tokenized_offsets = tokenized_offsets
        self._tokenized = tokenized
        self._eot_token_id = eot_token_id
        self._pad_token_id = pad_token_id
        self._source_language_tag_token_id = source_language_tag_token_id
        self._target_language_tag_token_id = target_language_tag_token_id
        self._combine_source_and_target = combine_source_and_target

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]

        x = self._tokenized[self._tokenized_offsets[idx]:self._tokenized_offsets[idx+1]]
        x = torch.from_numpy(x.astype(np.int64))

        source_end = torch.where(x == self._eot_token_id)[0][0].item()
        source = x[:source_end+1]
        target = x[source_end+1:]

        # adding a language tag to denote start of language text per "Language models are good translators", Wang et al
        x = torch.concatenate([torch.tensor([self._source_language_tag_token_id]), source, torch.tensor([self._target_language_tag_token_id]), target])

        source = x[:source_end+2]
        target = x[source_end+2:]

        if self._combine_source_and_target:
            combined_target = torch.cat([x[1:], torch.tensor([self._pad_token_id])])
        else:
            combined_target = None

        return source, target, x, combined_target, None


class SentencePairsDataset(Dataset):
    def __init__(
        self,
        datasets: LanguagePairsDatasets,
        idxs: np.ndarray,
        eos_token_id: int,
        pad_token_id: int,
        combined_tokenizer: SentencePieceTokenizer | Encoding,
        combine_source_and_target: bool = False,
        source_language_tag_token_id: Optional[int] = None,
        target_language_tag_token_id: Optional[int] = None,
        max_length: int = None,
    ):
        """

        :param datasets: LanguagePairsDatasets
        :param idxs: idxs to use from LanguagePairsDatasets
        :param tokenizer: SentencePieceTokenizer
        :param combine_source_and_target: Combine the source and target into a single input.
        :param max_length:
        :param eos_token_id
        """

        self._datasets = datasets
        self._idxs = idxs
        self._combined_tokenizer = combined_tokenizer
        self._max_length = max_length
        self._combine_source_and_target = combine_source_and_target
        self._source_language_tag_token_id = source_language_tag_token_id
        self._target_language_tag_token_id = target_language_tag_token_id
        self._transform = self._get_transform(max_len=max_length)
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]
        source, target, dataset_name = self._datasets[idx]

        if isinstance(self._combined_tokenizer, SentencePieceTokenizer):
            source_ids = self._combined_tokenizer.processor.encode(source)
            target_ids = self._combined_tokenizer.processor.encode(target)
        else:
            source_ids = self._combined_tokenizer.encode_ordinary(source)
            target_ids = self._combined_tokenizer.encode_ordinary(target)

        if self._combine_source_and_target:
            if self._source_language_tag_token_id is None or self._target_language_tag_token_id is None:
                raise ValueError('must provide token tags')
            # adding a language tag to denote start of language text per "Language models are good translators", Wang et al
            source = torch.concatenate([torch.tensor([self._source_language_tag_token_id]), self._transform(source_ids)])
            target = torch.concatenate([torch.tensor([self._target_language_tag_token_id]), self._transform(target_ids)])
        else:
            source = self._transform(source_ids)
            target = self._transform(target_ids)

        if self._combine_source_and_target:
            combined = torch.cat([source, target])
            combined_target = torch.cat([combined[1:], torch.tensor([self._pad_token_id])])   # shift 1 to the right
        else:
            combined = None
            combined_target = None
        return source, target, combined, combined_target, dataset_name

    def _get_transform(self, max_len: Optional[int] = None):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """

        def transform(x):
            if max_len is not None:
                x = x[:max_len]

            x.append(self._eos_token_id)
            x = torch.tensor(x)
            return x

        return transform

    @property
    def source_tokenizer(self) -> SentencePieceTokenizer:
        return self._source_tokenizer

    @property
    def target_tokenizer(self) -> SentencePieceTokenizer:
        return self._target_tokenizer

    @property
    def combined_tokenizer(self) -> SentencePieceTokenizer:
        return self._combined_tokenizer