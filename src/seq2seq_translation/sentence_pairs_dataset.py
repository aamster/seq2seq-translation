from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer, EOS_ID, PAD_ID,
)


class SentencePairsDataset(Dataset):
    def __init__(
        self,
        datasets: LanguagePairsDatasets,
        idxs: np.ndarray,
        source_tokenizer: Optional[SentencePieceTokenizer] = None,
        target_tokenizer: Optional[SentencePieceTokenizer] = None,
        combined_tokenizer: Optional[SentencePieceTokenizer] = None,
        combine_source_and_target: bool = False,
        max_length: int = None,
    ):
        """

        :param datasets: LanguagePairsDatasets
        :param idxs: idxs to use from LanguagePairsDatasets
        :param source_tokenizer: SentencePieceTokenizer
        :param target_tokenizer: SentencePieceTokenizer
        :param combined_tokenizer: If provided, will use this instead of a separate source_tokenizer and target_tokenizer
        :param combine_source_and_target: Combine the source and target into a single input.
        :param max_length:
        """

        if source_tokenizer is None and target_tokenizer is None and combined_tokenizer is None:
            raise ValueError('provide either source_tokenizer and target_tokenizer or combined_tokenizer')
        if source_tokenizer is not None and target_tokenizer is None or target_tokenizer is not None and source_tokenizer is None:
            raise ValueError('must provide both source_tokenizer and target_tokenizer')
        self._datasets = datasets
        self._idxs = idxs
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer
        self._combined_tokenizer = combined_tokenizer
        self._max_length = max_length
        self._combine_source_and_target = combine_source_and_target
        self._transform = self._get_transform(max_len=max_length)

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]
        source, target, dataset_name = self._datasets[idx]

        if self._combined_tokenizer is not None:
            source_ids = self._combined_tokenizer.processor.encode(source)
            target_ids = self._combined_tokenizer.processor.encode(target)
        else:
            source_ids = self._source_tokenizer.processor.encode(source)
            target_ids = self._target_tokenizer.processor.encode(target)

        source = self._transform(source_ids)
        target = self._transform(target_ids)

        if self._combine_source_and_target:
            combined = torch.cat([source, target])
            combined_target = torch.cat([combined[1:], torch.tensor([PAD_ID])])   # shift 1 to the right
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

            x.append(EOS_ID)
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