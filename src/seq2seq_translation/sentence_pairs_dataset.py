from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)


class SentencePairsDataset(Dataset):
    def __init__(
        self,
        datasets: LanguagePairsDatasets,
        idxs: np.ndarray,
        source_tokenizer: SentencePieceTokenizer,
        target_tokenizer: SentencePieceTokenizer,
        max_length: int = None,
    ):
        self._datasets = datasets
        self._idxs = idxs
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer
        self._max_length = max_length
        self._transform = self._get_transform(max_len=max_length)

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]
        source, target, dataset_name = self._datasets[idx]
        source_ids = self._source_tokenizer.processor.encode(source)
        target_ids = self._target_tokenizer.processor.encode(target)

        source = self._transform(source_ids)
        target = self._transform(target_ids)

        return source, target, dataset_name

    def _get_transform(self, max_len: Optional[int] = None):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """

        def transform(x):
            if max_len is not None:
                x = x[:max_len]

            x.append(self._source_tokenizer.processor.eos_id())
            x = torch.tensor(x)
            return x

        return transform

    @property
    def source_tokenizer(self) -> SentencePieceTokenizer:
        return self._source_tokenizer

    @property
    def target_tokenizer(self) -> SentencePieceTokenizer:
        return self._target_tokenizer
