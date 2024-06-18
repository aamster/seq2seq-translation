import os
from pathlib import Path
from typing import Optional, List

import numpy as np

from seq2seq_translation.datasets.europarl import Europarl
from seq2seq_translation.datasets.news_commentary import NewsCommentaryDataset


class LanguagePairsDatasets:
    """Collection of `LanguagePairsDataset`"""
    def __init__(
        self,
        out_dir: Path,
        source_lang: str,
        target_lang: str,
        sample_fracs: Optional[List[float]] = None
    ):
        if sample_fracs is not None:
            assert len(sample_fracs) == 2
        else:
            sample_fracs = [None, None]
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
        os.makedirs(source_tokenizer_path.parent, exist_ok=True)
        with open(source_tokenizer_path, 'wb') as f:
            for dataset in self._datasets:
                with open(dataset.source_path, 'rb') as ds_source_f:
                    for line in ds_source_f:
                        f.write(line)

    def create_target_tokenizer_train_set(self, target_tokenizer_path: Path):
        os.makedirs(target_tokenizer_path.parent, exist_ok=True)
        with open(target_tokenizer_path, 'wb') as f:
            for dataset in self._datasets:
                with open(dataset.target_path, 'rb') as ds_target_f:
                    for line in ds_target_f:
                        f.write(line)

    @property
    def target_paths(self) -> List[Path]:
        return [x.target_path for x in self._datasets]

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
            offset = dataset.target_index[idx]
            if idx == len(dataset)-1:
                continue
            else:
                input_length = dataset.target_index[idx+1] - offset
            if input_length > max_len:
                max_len = input_length
                max_len_idx = i
        return max_len_idx
