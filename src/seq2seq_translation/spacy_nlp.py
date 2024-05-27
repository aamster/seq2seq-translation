from typing import List, Tuple

import numpy as np
import spacy
import torch
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from tqdm import tqdm


class SpacyTokenizer:
    def __init__(
        self,
        spacy_model_name: str,
        text: List[str],
        max_len: int,
        min_freq: int = 1,
        specials: Tuple[str] = ('<pad>', '<sos>', '<eos>', '<unk>'),
        truncate: bool = True
    ):
        self._lang = spacy.load(spacy_model_name)
        self._vocab = build_vocab_from_iterator(
            self._get_tokens(text=text),
            min_freq=min_freq,
            specials=list(specials),
            special_first=True
        )
        self._vocab.set_default_index(self._vocab['<unk>'])
        self._transform = self._get_transform(max_len=max_len, truncate=truncate)
        self._itos = self._vocab.get_itos()
        self._stoi = self._vocab.get_stoi()

    def __call__(
        self,
        x: str,
        **kwargs
    ):
        return self.encode(x=x)

    @property
    def vocab(self):
        return self._vocab

    @property
    def lang(self):
        return self._lang

    def get_vocab(self):
        return self._stoi

    def encode(self, x: str) -> torch.tensor:
        tokens = self._tokenize_str(x)
        tokens = self._transform(tokens)
        return tokens

    def convert_ids_to_tokens(self, tokens: List[int]):
        return [self._itos[x] for x in tokens]

    def convert_tokens_to_ids(self, tokens: List[str]):
        if isinstance(tokens, str):
            return self._stoi.get(tokens, self._stoi['<unk>'])
        else:
            return [self._stoi.get(x, self._stoi['<unk>']) for x in tokens]

    def _tokenize_str(self, text: str):
        return [token.text for token in self._lang.tokenizer(text)]

    def _get_tokens(self, text: List[str]):
        for t in tqdm(text, total=len(text), desc='Getting tokens'):
            yield self._tokenize_str(text=t)

    def _get_transform(self, max_len: int, truncate: bool = True):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """
        transforms = []
        if truncate:
            transforms.append(T.Truncate(max_len))

        transforms += [
            T.VocabTransform(vocab=self._vocab),
            T.AddToken(self._vocab['<eos>'], begin=False),
            T.ToTensor()
        ]
        text_tranform = T.Sequential(*transforms)
        return text_tranform

    @property
    def pad_token(self):
        return '<pad>'

    @property
    def eos_token(self):
        return '<eos>'

    @property
    def unk_token(self):
        return '<unk>'

    @property
    def pad_token_id(self):
        return 0

    @property
    def itos(self):
        return self._itos


class SpacyEmbedding:
    def __init__(self, tokenizer: SpacyTokenizer):
        self._tokenizer = tokenizer
        self._embedding_dim = self._tokenizer.lang('hello').vector.shape[0]
        self._unk_vector = torch.randn(self._embedding_dim)
        _embeddings = self._build_embedding_matrix()

        # The following weirdness is to match the huggingface api for embeddings
        class _EmbedTokens:
            @property
            def weight(self):
                return _embeddings

        class _Encoder:
            @property
            def embed_tokens(self):
                return _EmbedTokens()

        self._encoder = _Encoder()

    def _build_embedding_matrix(self):
        embeddings = torch.zeros(len(self._tokenizer.get_vocab()), self._embedding_dim)
        for i, tok in enumerate(tqdm(self._tokenizer.itos, desc='Building spacy embedding matrix')):
            doc = self._tokenizer.lang(tok)
            if doc.has_vector:
                embeddings[i] = torch.tensor(doc.vector)
            else:
                embeddings[i] = self._unk_vector
        return embeddings

    @property
    def encoder(self):
        return self._encoder

    def get_input_embeddings(self):
        return self._encoder.embed_tokens