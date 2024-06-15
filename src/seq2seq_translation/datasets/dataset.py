import abc
import os
from pathlib import Path
from typing import Tuple


class LanguagePairsDataset(abc.ABC):
    def __init__(self, out_dir: str | Path):
        self._out_dir = out_dir
        self._len = None
        os.makedirs(out_dir, exist_ok=True)
        self.download()
        self._preprocess_dataset()
        self._source_index, self._target_index = self._index_files()

    @abc.abstractmethod
    def download(self, **kwargs):
        raise NotImplementedError

    def _preprocess_dataset(self):
        return

    @abc.abstractmethod
    def __getitem__(self, idx) -> Tuple[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def _create_index(filepath: Path):
        index = []
        offset = 0
        with open(filepath, 'r') as file:
            for line in file:
                index.append(offset)
                offset += len(line.encode('utf-8'))
        return index

    @abc.abstractmethod
    def _index_files(self):
        raise NotImplementedError

    @property
    def target_index(self):
        return self._target_index