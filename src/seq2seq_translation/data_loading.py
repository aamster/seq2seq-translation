from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
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
        target_vocab: Vocab,
        target_vocab_id_tokenizer_id_map: Dict[int, int],
        max_length: int = 60,
    ):
        self._data = data
        self._tokenizer = tokenizer
        self._tokenizer_itos = {v: k for k, v in tokenizer.get_vocab().items()}
        self._max_length = max_length
        self._target_vocab_stoi = target_vocab.get_stoi()
        self._target_vocab_id_tokenizer_id_map = target_vocab_id_tokenizer_id_map

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
        decoder_input = self._tokenizer(
            target,
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
        )

        source = source['input_ids']
        decoder_input = decoder_input['input_ids']

        # Remove batch dimension
        source = source.squeeze(0)
        decoder_input = decoder_input.squeeze(0)

        target = torch.tensor([self._target_vocab_stoi[x] for x in [self._tokenizer_itos[x.item()] for x in decoder_input]])

        return source, decoder_input, target

    @property
    def target_vocab_id_tokenizer_id_map(self) -> Dict[int, int]:
        """
        get the mapping between the new target vocab and the tokenizer vocab,
        so that we can call .decode using the tokenizer

        :return:
        """
        return self._target_vocab_id_tokenizer_id_map


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
        src_batch, trg_input_batch, target_batch = zip(*batch)
        src_batch_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        trg_input_batch_padded = pad_sequence(
            trg_input_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        target_batch_padded = pad_sequence(
            target_batch,
            batch_first=True,
            padding_value=self._pad_token_id
        )
        return src_batch_padded, trg_input_batch_padded, target_batch_padded


def get_target_vocab(
    data:  List[Tuple[str, ...]],
    tokenizer: PreTrainedTokenizer,
    min_freq: int = 1
):
    """
    construct a new vocabulary that just consists of the target tokens
    (the embedding matrix includes tokens for multiple languages)

    :return:
    """
    target = [x[1] for x in data]

    target = [tokenizer.encode(x) for x in target]
    target = [['<sos>'] + tokenizer.convert_ids_to_tokens(x) for x in target]

    vocab = build_vocab_from_iterator(
        iterator=target,
        min_freq=min_freq,
        specials=[tokenizer.pad_token, tokenizer.eos_token, tokenizer.unk_token, '<sos>']
    )

    new_vocab_id_tokenizer_id_map = {
        vocab.get_stoi()[x]: tokenizer.convert_tokens_to_ids(x)
        for x in vocab.get_stoi()
    }

    return vocab, new_vocab_id_tokenizer_id_map