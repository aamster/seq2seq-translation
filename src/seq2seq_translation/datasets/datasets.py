import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from seq2seq_translation.datasets.wmt14 import WMT14


class LanguagePairsDatasets:
    """Collection of `LanguagePairsDataset`"""
    def __init__(
        self,
        out_dir: Path,
        source_lang: str,
        target_lang: str,
        is_test: bool = False
    ):
        if is_test:
            self._datasets = [
                WMT14(
                    out_dir=out_dir,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    split='test'
                )
            ]
        else:
            self._datasets = [
                WMT14(
                    out_dir=out_dir,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    split='train'
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
                f.write(b'\n')

    def create_target_tokenizer_train_set(self, target_tokenizer_path: Path):
        if target_tokenizer_path.exists():
            return
        os.makedirs(target_tokenizer_path.parent, exist_ok=True)
        with open(target_tokenizer_path, 'wb') as f:
            for i in tqdm(range(len(self)), desc='Creating target tokenizer train set'):
                f.write(self[i][1].encode('utf-8'))
                f.write(b'\n')

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
