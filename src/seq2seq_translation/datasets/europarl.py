"""European Parliament Proceedings Parallel Corpus dataset
https://www.statmt.org/europarl/
"""
from pathlib import Path
from typing import Optional


from seq2seq_translation.datasets.dataset import LanguagePairsDataset, _download_and_extract, \
    _separate_single_language_file


class Europarl(LanguagePairsDataset):
    def __init__(self, out_dir: str | Path, source_lang: str, target_lang: str, sample_frac: Optional[float] = None):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_path = out_dir / f'{source_lang}-{target_lang}_{source_lang}.txt'
        self._target_path = out_dir / f'{source_lang}-{target_lang}_{target_lang}.txt'
        self._raw_out_path = out_dir / f'{source_lang}-{target_lang}.tsv'
        super().__init__(out_dir=out_dir, sample_frac=sample_frac)

    def download(self):
        url = f'https://www.statmt.org/europarl/v10/training/europarl-v10.{self._source_lang}-{self._target_lang}.tsv.gz'

        out_path = Path(self._out_dir) / f'{self._source_lang}-{self._target_lang}.tsv'
        if self._source_path.exists() and self._target_path.exists():
            print(f'{out_path} already exists')
            return

        print(f'Downloading Europarl dataset to {out_path}')

        _download_and_extract(url=url, gzip_path=Path(f'{out_path}.tsv.gz'), out_path=out_path)

    def _preprocess_dataset(self):
        if self._source_path.exists() and self._target_path.exists():
            return
        _separate_single_language_file(
            path=self._raw_out_path, source_path=self._source_path, target_path=self._target_path
        )

    def _index_files(self):
        print(f'Indexing {self._source_path}')
        source_index = self._create_index(filepath=self._source_path)

        print(f'Indexing {self._target_path}')
        target_index = self._create_index(filepath=self._target_path)
        assert len(source_index) == len(target_index)
        return source_index, target_index

    def __len__(self):
        return len(self._source_index_sampled)

    def __getitem__(self, idx):
        with open(self._source_path, 'r') as f:
            f.seek(self._source_index_sampled[idx])
            source = f.readline().strip()
            f.seek(0)
        with open(self._target_path, 'r') as f:
            f.seek(self._target_index_sampled[idx])
            target = f.readline().strip()
            f.seek(0)
        return source, target, 'europarl'

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def target_path(self) -> Path:
        return self._target_path