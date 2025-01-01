import numpy as np
import torch


class DataSplitter:
    def __init__(self, n_examples: int, train_frac: float, rng=None):
        self._n_examples = n_examples
        self._train_frac = train_frac
        self._rng = rng

    def split(self):
        print(f"{self._n_examples} pairs")

        idxs = np.arange(self._n_examples)

        if self._rng is not None:
            self._rng.shuffle(idxs)
        else:
            np.random.shuffle(idxs)

        n_train = int(self._n_examples * self._train_frac)

        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        return train_idxs, test_idxs


class CollateFunction:
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        src_lengths = [len(x[0]) for x in batch]

        src_batch, target_batch, combined_batch, combined_target_batch, dataset_name = zip(*batch)
        src_batch_padded = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=self._pad_token_id
        )
        target_batch_padded = torch.nn.utils.rnn.pad_sequence(
            target_batch, batch_first=True, padding_value=self._pad_token_id
        )

        if all(x is None for x in combined_batch):
            combined_batch_padded = None
            combined_target_batch_padded = None
        else:
            combined_batch_padded = torch.nn.utils.rnn.pad_sequence(
                combined_batch, batch_first=True, padding_value=self._pad_token_id
            )
            combined_target_batch_padded = torch.nn.utils.rnn.pad_sequence(
                combined_target_batch, batch_first=True, padding_value=self._pad_token_id
            )

        return src_batch_padded, target_batch_padded, combined_batch_padded, combined_target_batch_padded, dataset_name, src_lengths
