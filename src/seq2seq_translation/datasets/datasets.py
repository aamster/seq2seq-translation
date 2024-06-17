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
        return dataset[idx]

    def __len__(self):
        return sum([len(x) for x in self._datasets])

    def _get_dataset_for_idx(self, idx: int):
        """
        Gets the dataset corresponding to `idx`
        
        :param idx:
        :return:
        """
        start = 0
        for i in range(len(self._datasets)):
            if idx <= start + len(self._datasets[i]):
                return self._datasets[i]
            else:
                start += len(self._datasets[i])
        else:
            raise RuntimeError(f'idx {idx} out of bounds')

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
            offset = dataset.target_index[idx]
            if idx == len(dataset)-1:
                input_length = len(dataset[len(dataset.target_index)-1][1])
            else:
                input_length = dataset.target_index[idx+1] - offset
            if input_length > max_len:
                max_len = input_length
                max_len_idx = i
        return max_len_idx
