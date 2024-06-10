from pathlib import Path

from seq2seq_translation.datasets.europarl import Europarl


class LanguagePairsDatasets:
    """Collection of `LanguagePairsDataset`"""
    def __init__(
        self,
        out_dir: Path,
        source_lang: str,
        target_lang: str
    ):
        self._datasets = [
            Europarl(
                out_dir=out_dir / 'europarl',
                source_lang=source_lang,
                target_lang=target_lang
            )
        ]

    def __getitem__(self, idx):
        start = 0
        for i in range(len(self._datasets)):
            if idx <= start + len(self._datasets[i]):
                return self._datasets[i][idx]
            else:
                start += len(self._datasets[i])
        else:
            raise RuntimeError(f'idx {idx} out of bounds')

    def __len__(self):
        return sum([len(x) for x in self._datasets])
