"""WMT 14 test set
https://www.statmt.org/wmt14/test-full.tgz
"""
import importlib
import site
import sys
from pathlib import Path
from typing import Optional


from seq2seq_translation.datasets.dataset import LanguagePairsDataset

# need to do it this way due to name clash with local "datasets" package
sys.path.insert(0, site.getsitepackages()[0])
from datasets import load_dataset
sys.path.pop(0)


class WMT14_Test(LanguagePairsDataset):
    def __init__(self, out_dir: str | Path, source_lang: str, target_lang: str):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._ds = None
        super().__init__(out_dir=out_dir, sample_frac=None)

    def download(self):
        ds = load_dataset("wmt/wmt14", f"{self._source_lang}-{self._target_lang}", split='test',
                     cache_dir=str(self._out_dir))
        self._ds = ds

    def _preprocess_dataset(self):
        return

    def _index_files(self):
        return None, None

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        source = self._ds['translation'][idx][self._source_lang]
        target = self._ds['translation'][idx][self._target_lang]
        return source, target, 'wmt14_test'

    @property
    def source_path(self) -> Optional[Path]:
        return None

    @property
    def target_path(self) -> Optional[Path]:
        return None
