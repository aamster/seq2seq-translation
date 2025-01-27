import tiktoken
import torch


class TikTokenTokenizer:
    _tokenizer = tiktoken.get_encoding('gpt2')

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def pad_idx(self) -> int:
        return self._tokenizer.max_token_value + 2 # 1 more than eot token

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab + 2  # includes eot and pad ids

    def decode(self, token_ids: torch.tensor) -> list[str] | str:
        """
        Decode `token_ids` to a string or list of strings
        Each list of token_ids is truncated to the first occurance of the eos token

        :param token_ids: token ids to decode
        :return:
        """
        decoded = []
        if len(token_ids.shape) == 1:
            token_ids = token_ids.reshape(1, -1)
        for tokens in token_ids:
            eos_idx = torch.where(tokens == self._tokenizer.eot_token)[0]
            if len(eos_idx) > 0:
                eos_idx = eos_idx[0]
                tokens = tokens[:eos_idx]
            tokens = tokens[tokens != self.pad_idx]
            decoded.append(self._tokenizer.decode(tokens.tolist()))
        if len(decoded) == 1:
            decoded = decoded[0]
        return decoded
