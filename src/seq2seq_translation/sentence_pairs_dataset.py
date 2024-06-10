from typing import List, Tuple, Dict, Optional

import numpy as np
from torch.utils.data import Dataset
import torchtext.transforms as T

from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer


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
        source, target = self._datasets[idx]
        source_ids = self._source_tokenizer.processor.encode(source)
        target_ids = self._target_tokenizer.processor.encode(target)

        source = self._transform(source_ids)
        target = self._transform(target_ids)

        return source, target

    def _get_transform(self, max_len: Optional[int] = None):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """
        transforms = []
        if max_len is not None:
            transforms.append(T.Truncate(max_len))

        transforms += [
            T.AddToken(self._source_tokenizer.processor.bos_id(), begin=True),
            T.AddToken(self._source_tokenizer.processor.eos_id(), begin=False),
            T.ToTensor()
        ]
        text_tranform = T.Sequential(*transforms)
        return text_tranform
