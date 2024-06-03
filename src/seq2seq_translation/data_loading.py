import re
import unicodedata
from typing import List, Tuple

import numpy as np
from torch.nn.utils.rnn import pad_sequence


def _preprocess_string(
    s: str,
    lowercase: bool = False,
    remove_diacritical_marks: bool = False,
    remove_non_eos_punctuation: bool = False
):
    if lowercase:
        s = s.lower()
    if remove_diacritical_marks:
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    if remove_non_eos_punctuation:
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s


def read_data(
    data_path: str,
    source_lang: str,
    target_lang: str
):
    print("Reading lines...")

    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')

    pairs: List[Tuple[str, ...]] = [
        tuple([
            _preprocess_string(
                s=s,
                lowercase=False,
                remove_diacritical_marks=False,
                remove_non_eos_punctuation=False
            )
            for s in l.split('\t')]) for l in lines]

    if source_lang == 'en' and target_lang == 'fr':
        pass
    elif source_lang == 'fr' and target_lang == 'en':
        pairs = [tuple(reversed(x)) for x in pairs]
    else:
        raise NotImplementedError
    return pairs


class DataSplitter:
    def __init__(self, data: List[Tuple], train_frac: float):
        self._data = data
        self._train_frac = train_frac

    def split(self):
        print(f'{len(self._data)} pairs')

        idxs = np.arange(len(self._data))
        np.random.shuffle(idxs)

        n_train = int(len(self._data) * self._train_frac)

        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        train = [self._data[x] for x in train_idxs]
        test = [self._data[x] for x in test_idxs]
        return train, test


class CollateFunction:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        src_batch, target_batch = zip(*batch)
        src_batch_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        target_batch_padded = pad_sequence(
            target_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        return src_batch_padded, target_batch_padded
