import gzip
import os
import shutil
from pathlib import Path
from typing import Tuple

import requests

from seq2seq_translation.datasets.dataset import LanguagePairsDataset


class NewsCommentaryDataset(LanguagePairsDataset):
    """
    https://data.statmt.org/news-commentary/README
    """
    def __init__(self, out_dir: str | Path, source_lang: str, target_lang: str):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_path = out_dir / f'{source_lang}-{target_lang}.{source_lang}'
        self._target_path = out_dir / f'{source_lang}-{target_lang}.{target_lang}'
        self._raw_out_path = out_dir / f'{source_lang}-{target_lang}'
        super().__init__(out_dir=out_dir)

    def _preprocess_dataset(self):
        if self._source_path.exists() and self._target_path.exists():
            return
        print('Preprocessing News Commentary dataset')
        source_lines = []
        target_lines = []
        with open(self._raw_out_path) as f:
            for line in f:
                if line.strip():
                    source, target = line.split('\t')
                    source_lines.append(source+'\n')
                    target_lines.append(target)

        with open(self._source_path, 'w') as f:
            for line in source_lines:
                f.write(line)

        with open(self._target_path, 'w') as f:
            for line in target_lines:
                f.write(line)

        os.remove(self._raw_out_path)

    def download(self, **kwargs):
        url = f'https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.{self._source_lang}-{self._target_lang}.tsv.gz'

        out_path = Path(self._out_dir) / f'{self._source_lang}-{self._target_lang}'
        if self._source_path.exists() and self._target_path.exists():
            print(f'News Commentary dataset has already been downloaded')
            return

        print(f'Downloading News Commentary dataset to {out_path}')

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(f'{out_path}.tsv.gz', "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with gzip.open(f'{out_path}.tsv.gz', 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(f'{out_path}.tsv.gz')

    def __getitem__(self, idx) -> Tuple[str, str]:
        with open(self._source_path, 'r') as f:
            f.seek(self._source_index[idx])
            source = f.readline()
            f.seek(0)
        with open(self._target_path, 'r') as f:
            f.seek(self._target_index[idx])
            target = f.readline()
            f.seek(0)
        return source, target

    def __len__(self):
        return len(self._source_index)

    def _index_files(self):
        print(f'Indexing {self._source_path}')
        source_index = self._create_index(filepath=self._source_path)

        print(f'Indexing {self._target_path}')
        target_index = self._create_index(filepath=self._target_path)
        return source_index, target_index
