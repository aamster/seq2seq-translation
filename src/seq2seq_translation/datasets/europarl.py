"""European Parliament Proceedings Parallel Corpus dataset
https://www.statmt.org/europarl/
"""
import tarfile
from pathlib import Path

import requests

from seq2seq_translation.datasets.dataset import LanguagePairsDataset


class Europarl(LanguagePairsDataset):
    def __init__(self, out_dir: str | Path, source_lang: str, target_lang: str):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_path = (out_dir / f'{source_lang}-{target_lang}' /
                             f'europarl-v7.{source_lang}-{target_lang}.'
                             f'{source_lang}')
        self._target_path = (out_dir / f'{source_lang}-{target_lang}' /
                             f'europarl-v7.{source_lang}-{target_lang}.'
                             f'{target_lang}')
        super().__init__(out_dir=out_dir)

    def download(self):
        url = f'https://www.statmt.org/europarl/v7/{self._source_lang}-{self._target_lang}.tgz'

        out_path = Path(self._out_dir) / f'{self._source_lang}-{self._target_lang}'
        if out_path.exists():
            print(f'{out_path} already exists')
            return

        print(f'Downloading Europarl dataset to {out_path}')

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(f'{out_path}.tgz', "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with tarfile.open(f'{out_path}.tgz', "r:gz") as tar:
            tar.extractall(path=out_path)

    def _index_files(self):
        print(f'Indexing {self._source_path}')
        source_index = self._create_index(filepath=self._source_path)

        print(f'Indexing {self._target_path}')
        target_index = self._create_index(filepath=self._target_path)
        assert len(source_index) == len(target_index)
        return source_index, target_index

    def __len__(self):
        return len(self._source_index)

    def __getitem__(self, idx):
        with open(self._source_path, 'r') as f:
            f.seek(self._source_index[idx])
            source = f.readline()
            f.seek(0)
        with open(self._target_path, 'r') as f:
            f.seek(self._target_index[idx])
            target = f.readline()
            f.seek(0)
        return source, target
