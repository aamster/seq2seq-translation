import gzip
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from seq2seq_translation.datasets.dataset import LanguagePairsDataset, _download_and_extract, \
    _separate_single_language_file


class NewsCommentaryDataset(LanguagePairsDataset):
    """
    https://data.statmt.org/news-commentary/README
    """
    def __init__(self, out_dir: str | Path, source_lang: str, target_lang: str, sample_frac: Optional[float] = None):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_path = out_dir / f'{source_lang}-{target_lang}.{source_lang}'
        self._target_path = out_dir / f'{source_lang}-{target_lang}.{target_lang}'
        self._raw_out_path = out_dir / f'{source_lang}-{target_lang}.tsv'
        super().__init__(out_dir=out_dir, sample_frac=sample_frac)

    def _preprocess_dataset(self):
        if self._source_path.exists() and self._target_path.exists():
            return
        print('Preprocessing News Commentary dataset')
        _separate_single_language_file(
            path=self._raw_out_path, source_path=self._source_path, target_path=self._target_path
        )

    def download(self, **kwargs):
        url = f'https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.{self._source_lang}-{self._target_lang}.tsv.gz'

        out_path = Path(self._out_dir) / f'{self._source_lang}-{self._target_lang}.tsv'
        if self._source_path.exists() and self._target_path.exists():
            print(f'News Commentary dataset has already been downloaded')
            return

        print(f'Downloading News Commentary dataset to {out_path}')

        _download_and_extract(url=url, gzip_path=Path(f'{out_path}.tsv.gz'), out_path=out_path)

    def __getitem__(self, idx) -> Tuple[str, str, str]:
        with open(self._source_path, 'r') as f:
            f.seek(self._source_index_sampled[idx])
            source = f.readline().strip()
            f.seek(0)
        with open(self._target_path, 'r') as f:
            f.seek(self._target_index_sampled[idx])
            target = f.readline().strip()
            f.seek(0)

        # note: reversing, since we want *->en rather than en->*
        return target, source, 'news commentary'

    def __len__(self):
        return len(self._source_index_sampled)

    @property
    def target_index(self):
        # swapping bc raw data stored as en-fr so target, en is actually source index
        # TODO clean this up
        return self._source_index

    def _index_files(self):
        print(f'Indexing {self._source_path}')
        source_index = self._create_index(filepath=self._source_path)

        print(f'Indexing {self._target_path}')
        target_index = self._create_index(filepath=self._target_path)
        return source_index, target_index

    @property
    def source_path(self) -> Path:
        # Swapping bc most datasets are en-* and we want en to be target
        return self._target_path

    @property
    def target_path(self) -> Path:
        # Swapping bc most datasets are en-* and we want en to be target
        return self._source_path
