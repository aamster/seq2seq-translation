"""WMT 14 dataset
https://www.statmt.org/wmt14/test-full.tgz
"""

import json
import site
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from seq2seq_translation.datasets.dataset import LanguagePairsDataset

# need to do it this way due to name clash with local "datasets" package
sys.path.insert(0, site.getsitepackages()[0])
from datasets import load_dataset

sys.path.pop(0)


class WMT14(LanguagePairsDataset):
    def __init__(
        self, out_dir: str | Path, source_lang: str, target_lang: str, split: str
    ):
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._ds = None
        self._split = split
        self._source_path = Path(out_dir) / f"{self._source_lang}.txt"
        self._target_path = Path(out_dir) / f"{self._target_lang}.txt"
        self._index_path = Path(out_dir) / f"indexes.json"
        self._streaming = split != "test"
        super().__init__(out_dir=out_dir, sample_frac=None)

    def download(self):
        ds_name = (
            f"{self._source_lang}-{self._target_lang}"
            if self._target_lang == "en"
            else f"{self._target_lang}-{self._source_lang}"
        )
        self._ds = load_dataset(
            "wmt/wmt14",
            ds_name,
            split=self._split,
            cache_dir=str(self._out_dir),
            streaming=self._streaming,
        )

    def write_to_single_file(self):
        if self._source_path.exists() and self._target_path.exists():
            return

        source_offset = 0
        target_offset = 0

        offsets = {"source": [], "target": []}
        with open(self._source_path, "wb") as source_f:
            with open(self._target_path, "wb") as target_f:
                for example in tqdm(
                    iter(self._ds),
                    desc=f"Writing WMT14 {self._split} to disk",
                    total=len(self),
                ):
                    source = (example["translation"][self._source_lang] + "\n").encode(
                        "utf-8"
                    )
                    target = (example["translation"][self._target_lang] + "\n").encode(
                        "utf-8"
                    )
                    source_f.write(source)
                    target_f.write(target)

                    offsets["source"].append(source_offset)
                    offsets["target"].append(target_offset)

                    source_offset += len(source)
                    target_offset += len(target)

        with open(self._index_path, "w") as f:
            f.write(json.dumps(offsets))

    def _preprocess_dataset(self):
        return

    def _index_files(self):
        if self._split == "test":
            return None, None
        else:
            with open(self._index_path) as f:
                index = json.load(f)

            if self._source_lang == "en":
                # swapping because en was indexed as source
                return index["target"], index["source"]
            else:
                return index["source"], index["target"]

    def __len__(self):
        return self._ds.info.splits[self._split].num_examples

    def __getitem__(self, idx):
        if self._split == "test":
            # load it from the huggingface api
            source = self._ds[int(idx)]["translation"][self._source_lang]
            target = self._ds[int(idx)]["translation"][self._target_lang]

            if idx == 729:
                # it contains corrupt text
                # Replace the misencoded character with the correct EN DASH (U+2013)
                source = source.replace("\x96", "\u2013")
        else:
            # load it from on disk (for random access)
            with open(self._source_path, "r", encoding="utf-8") as f:
                f.seek(self._source_index_sampled[idx])
                source = f.readline().strip()
                f.seek(0)
            with open(self._target_path, "r", encoding="utf-8") as f:
                f.seek(self._target_index_sampled[idx])
                target = f.readline().strip()
                f.seek(0)
        return source, target, f"wmt14_{self._split}"

    @property
    def source_path(self) -> Optional[Path]:
        return None

    @property
    def target_path(self) -> Optional[Path]:
        return None
