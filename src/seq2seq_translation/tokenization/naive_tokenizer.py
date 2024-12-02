import re
from typing import List, Tuple

import torch
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from tqdm import tqdm


class NaiveTokenizer:
    """
    Tokenizes text by splitting on space
    """

    def __init__(
        self,
        text: List[str],
        max_len: int,
        min_freq: int = 1,
        specials: Tuple[str] = ("<pad>", "<sos>", "<eos>", "<unk>"),
        truncate: bool = True,
    ):
        self._vocab = build_vocab_from_iterator(
            self._get_tokens(text=text),
            min_freq=min_freq,
            specials=list(specials),
            special_first=True,
        )
        self._vocab.set_default_index(self._vocab["<unk>"])
        self._transform = self._get_transform(max_len=max_len, truncate=truncate)
        self._itos = self._vocab.get_itos()
        self._stoi = self._vocab.get_stoi()

    def __call__(self, x: str, **kwargs):
        return self.encode(x=x)

    @property
    def vocab(self):
        return self._vocab

    def get_vocab(self):
        return self._stoi

    def encode(self, x: str) -> torch.tensor:
        tokens = self._tokenize_str(x)
        tokens = self._transform(tokens)
        return tokens

    def convert_ids_to_tokens(self, tokens: List[int]):
        return [self._itos[x] for x in tokens]

    def convert_tokens_to_ids(self, tokens: List[str] | str):
        if isinstance(tokens, str):
            return self._stoi.get(tokens, self._stoi["<unk>"])
        else:
            return [self._stoi.get(x, self._stoi["<unk>"]) for x in tokens]

    def _tokenize_str(self, text: str):
        text = re.sub(r"([.!?])", r" \1", text)
        tokens = [x for x in text.split(" ") if len(x) > 0]
        return tokens

    def _get_tokens(self, text: List[str]):
        for t in tqdm(text, total=len(text), desc="Getting tokens"):
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
            T.AddToken(self._vocab["<eos>"], begin=False),
            T.ToTensor(),
        ]
        text_tranform = T.Sequential(*transforms)
        return text_tranform

    @property
    def pad_token(self):
        return "<pad>"

    @property
    def eos_token(self):
        return "<eos>"

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token_id(self):
        return 0

    @property
    def itos(self):
        return self._itos
