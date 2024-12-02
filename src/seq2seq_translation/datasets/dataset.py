import abc
import gzip
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import requests


class LanguagePairsDataset(abc.ABC):
    def __init__(self, out_dir: str | Path, sample_frac: Optional[float] = None):
        self._out_dir = out_dir
        self._len = None
        self._sample_frac = sample_frac
        os.makedirs(out_dir, exist_ok=True)
        self.download()
        self._preprocess_dataset()
        self._source_index, self._target_index = self._index_files()

        if sample_frac is not None:
            self._source_index_sampled, self._target_index_sampled = self._sample()
        else:
            self._source_index_sampled = self._source_index
            self._target_index_sampled = self._target_index

    @abc.abstractmethod
    def download(self, **kwargs):
        raise NotImplementedError

    def _preprocess_dataset(self):
        return

    @abc.abstractmethod
    def __getitem__(self, idx) -> Tuple[str, str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def _create_index(filepath: Path):
        index = []
        offset = 0
        with open(filepath, "r") as file:
            for i, line in enumerate(file):
                index.append(offset)
                offset += len(line.encode("utf-8"))
        return index

    @abc.abstractmethod
    def _index_files(self):
        raise NotImplementedError

    @property
    def target_index(self):
        return self._target_index

    def _sample(self):
        idxs = np.arange(len(self._source_index))
        np.random.shuffle(idxs)
        idxs = idxs[: int(len(idxs) * self._sample_frac)]
        source_index = [self._source_index[idx] for idx in idxs]
        target_index = [self._target_index[idx] for idx in idxs]

        print(
            f"Number of examples for {type(self)} after sampling {self._sample_frac}: {len(source_index)}"
        )
        return source_index, target_index


def _download_and_extract(url: str, gzip_path: Path, out_path: Path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(gzip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    with gzip.open(gzip_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(f"{out_path}.tsv.gz")


def _separate_single_language_file(path: Path, source_path: Path, target_path: Path):
    # TODO we need to remove empty lines
    source_lines = []
    target_lines = []
    with open(path) as f:
        for line in f:
            if line.strip():
                split = line.split("\t")
                # europarl adds additional fields after source, target
                split = split[:2]
                source, target = split
                source_lines.append(source + "\n")
                if len(target) == 0:
                    target = "\n"
                if target[-1] != "\n":
                    target += "\n"
                target_lines.append(target)

    with open(source_path, "w") as f:
        for line in source_lines:
            f.write(line)

    with open(target_path, "w") as f:
        for line in target_lines:
            f.write(line)

    os.remove(path)
