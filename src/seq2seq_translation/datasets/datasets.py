import os
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from seq2seq_translation.datasets.europarl import Europarl
from seq2seq_translation.datasets.news_commentary import NewsCommentaryDataset
from seq2seq_translation.datasets.wmt14_test import WMT14_Test


class LanguagePairsDatasets:
    """Collection of `LanguagePairsDataset`"""
    def __init__(
        self,
        out_dir: Path,
        source_lang: str,
        target_lang: str,
        sample_fracs: Optional[List[float]] = None,
        is_test: bool = False
    ):
        if sample_fracs is not None:
            assert len(sample_fracs) == 2
        else:
            sample_fracs = [None, None]
        if is_test:
            self._datasets = [
                WMT14_Test(
                    out_dir=out_dir / 'wmt14_test',
                    source_lang=source_lang,
                    target_lang=target_lang
                )
            ]
        else:
            self._datasets = [
                Europarl(
                    out_dir=out_dir / 'europarl',
                    source_lang=source_lang,
                    target_lang=target_lang,
                    sample_frac=sample_fracs[0]
                ),
                NewsCommentaryDataset(
                    out_dir=out_dir / 'news_commentary',
                    # swapping bc most datasets are en-*
                    source_lang=target_lang,
                    target_lang=source_lang,
                    sample_frac=sample_fracs[1]
                )
            ]

    def __getitem__(self, idx):
        dataset = self._get_dataset_for_idx(idx=idx)
        idx = self._get_dataset_index(idx=idx)
        return dataset[idx]

    def __len__(self):
        return sum([len(x) for x in self._datasets])

    def create_source_tokenizer_train_set(self, source_tokenizer_path: Path):
        if source_tokenizer_path.exists():
            return
        os.makedirs(source_tokenizer_path.parent, exist_ok=True)
        with open(source_tokenizer_path, 'wb') as f:
            for i in tqdm(range(len(self)), desc='Creating source tokenizer train set'):
                f.write(self[i][0].encode('utf-8'))

    def create_target_tokenizer_train_set(self, target_tokenizer_path: Path):
        if target_tokenizer_path.exists():
            return
        os.makedirs(target_tokenizer_path.parent, exist_ok=True)
        with open(target_tokenizer_path, 'wb') as f:
            for i in tqdm(range(len(self)), desc='Creating target tokenizer train set'):
                f.write(self[i][1].encode('utf-8'))

    def _get_dataset_for_idx(self, idx: int):
        """
        Gets the dataset corresponding to `idx`
        
        :param idx:
        :return:
        """
        start = 0
        for i in range(len(self._datasets)):
            if idx < start + len(self._datasets[i]):
                return self._datasets[i]
            else:
                start += len(self._datasets[i])
        else:
            raise RuntimeError(f'idx {idx} out of bounds')

    def _get_dataset_index(self, idx: int):
        """Makes sure that the index starts at 0 for each dataset
        e.g. idx = 150
        dataset 0 has len 100
        dataset 1 has len 200

        The new index should be 50 for dataset 1
        """
        dataset = self._get_dataset_for_idx(idx=idx)
        dataset_index = [i for i in range(len(self._datasets)) if self._datasets[i] == dataset][0]
        if dataset_index > 0:
            for i in range(dataset_index):
                idx -= len(self._datasets[i])
        return idx

    def get_max_target_length_index(self, from_indexes: np.ndarray) -> int:
        """
        Gets the argmax of the examples in the targets

        :param from_indexes: Indices to choose from
        :return:
        """
        max_len = 0
        max_len_idx = None

        for i, idx in enumerate(from_indexes):
            dataset = self._get_dataset_for_idx(idx=idx)
            idx = self._get_dataset_index(idx=idx)
            offset_start = dataset.target_index[idx]
            if idx == len(dataset)-1:
                continue
            else:
                offset_end = dataset.target_index[idx+1]
                input_length = offset_end - offset_start
            if input_length > max_len:
                max_len = input_length
                max_len_idx = idx
        return max_len_idx
