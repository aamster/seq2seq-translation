from typing import List, Tuple

import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def read_data(data_path: str):
    print("Reading lines...")

    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')

    pairs: List[Tuple[str, ...]] = [tuple([s.strip() for s in l.split('\t')]) for l in lines]

    return pairs


class SentencePairsDataset(Dataset):
    def __init__(
        self,
        data: List[Tuple[str, ...]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 60
    ):
        self._data = data
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        source, target = self._data[idx]
        source = self._tokenizer(
            source,
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
        )
        target = self._tokenizer(
            target,
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
        )

        source = source['input_ids']
        target = target['input_ids']

        # Remove batch dimension
        source = source.squeeze(0)
        target = target.squeeze(0)

        return source, target


class DataSplitter:
    def __init__(self, data_path: str, train_frac: float):
        self._data_path = data_path
        self._train_frac = train_frac

    def split(self):
        data = read_data(data_path=self._data_path)
        print(f'{len(data)} pairs')

        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        n_train = int(len(data) * self._train_frac)

        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        train = [data[x] for x in train_idxs]
        test = [data[x] for x in test_idxs]
        return train, test


class CollateFunction:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        trg_batch_padded = pad_sequence(
            trg_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        return src_batch_padded, trg_batch_padded
