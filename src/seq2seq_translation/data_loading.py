from typing import List, Tuple, Optional

import numpy as np
import spacy
from spacy import Language
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torchtext.transforms as T

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def _normalize_string(s) -> str:
    s = s.lower().strip()
    return s


def read_data(data_path: str):
    print("Reading lines...")

    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')

    pairs: List[Tuple[str, ...]] = [tuple([_normalize_string(s) for s in l.split('\t')]) for l in lines]

    return pairs


def _tokenize_str(spacy_lang: Language, text: str):
    return [token.text for token in spacy_lang.tokenizer(text)]


def get_tokens(spacy_lang: Language, text: List[str]):
    for t in text:
        yield _tokenize_str(spacy_lang=spacy_lang, text=t)


def get_vocabs(text_pairs: List[Tuple[str]],
               source_spacy_language_model_name: str,
               target_spacy_language_model_name: str):
    source_lang = spacy.load(source_spacy_language_model_name)
    target_lang = spacy.load(target_spacy_language_model_name)

    specials = [''] * 4
    specials[PAD_IDX] = '<pad>'
    specials[BOS_IDX] = '<sos>'
    specials[EOS_IDX] = '<eos>'
    specials[UNK_IDX] = '<unk>'

    source_vocab = build_vocab_from_iterator(
        get_tokens(spacy_lang=source_lang, text=[x[0] for x in text_pairs]),
        min_freq=2,
        specials=specials,
        special_first=True
    )
    source_vocab.set_default_index(source_vocab[specials[UNK_IDX]])

    target_vocab = build_vocab_from_iterator(
        get_tokens(spacy_lang=target_lang, text=[x[1] for x in text_pairs]),
        min_freq=2,
        specials=specials,
        special_first=True
    )
    target_vocab.set_default_index(target_vocab[specials[UNK_IDX]])
    return source_vocab, target_vocab


def get_transform(vocab: Vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        T.AddToken(BOS_IDX, begin=True),
        T.AddToken(EOS_IDX, begin=False),
        T.ToTensor(padding_value=PAD_IDX)
    )
    return text_tranform


class SentencePairsDataset(Dataset):
    def __init__(
            self,
            data: List[Tuple[str, ...]],
            source_spacy_lang: str,
            target_spacy_lang: str,
            source_transform: Optional[T.Sequential] = None,
            target_transform: Optional[T.Sequential] = None):
        self._data = data
        self._source_transform = source_transform
        self._target_transform = target_transform
        self._source_spacy_lang = spacy.load(source_spacy_lang)
        self._target_spacy_lang = spacy.load(target_spacy_lang)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        source, target = self._data[idx]
        source = _tokenize_str(spacy_lang=self._source_spacy_lang, text=source)
        target = _tokenize_str(spacy_lang=self._source_spacy_lang, text=target)
        if self._source_transform is not None:
            source = self._source_transform([source])
        if self._target_transform is not None:
            target = self._target_transform(target)

        return source, target


class DataSplitter:
    def __init__(self, data_path: str, train_frac: float, max_len: Optional[int] = None):
        self._data_path = data_path
        self._train_frac = train_frac
        self._max_len = max_len

    def split(self):
        data = read_data(data_path=self._data_path)
        print(f'{len(data)} pairs')

        if self._max_len is not None:
            data = [x for x in data if len(x[0]) < self._max_len]
            print(f'Filtering to {len(data)} pairs')

        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        n_train = int(len(data) * self._train_frac)

        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        train = [data[x] for x in train_idxs]
        test = [data[x] for x in test_idxs]
        return train, test


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample.reshape(-1))
        tgt_batch.append(tgt_sample.reshape(-1))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch
