import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from tiktoken import Encoding
from tqdm import tqdm
import numpy as np
import tiktoken

from seq2seq_translation.data_loading import DataSplitter
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)


@dataclass
class Config:
    datasets_dir: Path
    source_lang: str
    target_lang: str
    seed: int
    out_dir: Path
    train_frac: float = 0.999
    tokenizer_type: str = "tiktoken"
    sentence_piece_model_dir: Optional[Path] = None
    max_len: int = 128
    train_n_tokens: Optional[int] = None
    val_n_tokens: Optional[int] = None
    batch_size: int = 2**18
    num_workers: int = os.cpu_count()

    def __post_init__(self):
        self.datasets_dir = Path(self.datasets_dir)
        self.out_dir = Path(self.out_dir)

        if self.sentence_piece_model_dir is not None:
            self.sentence_piece_model_dir = Path(self.sentence_piece_model_dir)


def process(source, target, enc: Encoding | SentencePieceTokenizer, max_len: int = 128):

    if isinstance(enc, Encoding):
        source_ids = enc.encode_ordinary(source)
        target_ids = enc.encode_ordinary(target)
    else:
        source_ids = enc.processor.encode(source)
        target_ids = enc.processor.encode(target)

    source_ids = source_ids[:max_len]
    target_ids = target_ids[:max_len]

    if isinstance(enc, Encoding):
        source_ids.append(enc.eot_token)
        target_ids.append(enc.eot_token)
    else:
        source_ids.append(enc.eot_idx)
        target_ids.append(enc.eot_idx)

    combined = source_ids + target_ids

    return combined


def tokenize(
    batch_idx: int,
    enc: Encoding | SentencePieceTokenizer,
    datasets,
    dataset_len: int,
    batch_size: int = 1024,
):
    start = batch_idx * batch_size
    end = min(start + batch_size, dataset_len)
    batch = [process(*datasets[i][:-1], enc=enc) for i in range(start, end)]
    return batch, batch_idx


def get_num_tokens_parallel(
    enc,
    datasets: LanguagePairsDatasets,
    idxs: np.ndarray,
    batch_size: int,
    num_batches: int,
    num_workers: int,
):
    tokenize_partial = partial(
        tokenize,
        enc=enc,
        datasets=datasets,
        dataset_len=len(idxs),
        batch_size=batch_size,
    )

    num_tokens = 0

    batch_lens = np.zeros((num_batches,), dtype=np.int64)

    with Pool(num_workers) as pool:
        with tqdm(total=num_batches, desc="Getting num tokens") as pbar:
            for result in pool.imap_unordered(tokenize_partial, range(num_batches)):
                batch, batch_idx = result
                batch_num_tokens = sum([len(x) for x in batch])
                num_tokens += batch_num_tokens
                batch_lens[batch_idx] = batch_num_tokens
                pbar.update(1)
    return num_tokens, batch_lens


def write_tokens_to_memmap_parallel(
    enc,
    datasets: LanguagePairsDatasets,
    idxs: np.ndarray,
    batch_size: int,
    num_batches: int,
    arr: np.memmap,
    batch_lens: np.ndarray,
    num_workers: int,
    offsets_out_path: Path,
):
    tokenize_partial = partial(
        tokenize,
        enc=enc,
        datasets=datasets,
        dataset_len=len(idxs),
        batch_size=batch_size,
    )

    offsets = np.memmap(
        offsets_out_path, dtype=np.uint64, mode="w+", shape=(len(idxs) + 1,)
    )
    offsets[0] = 0

    with Pool(num_workers) as pool:
        with tqdm(total=num_batches, desc="Writing to memmap") as pbar:
            for result in pool.imap(tokenize_partial, range(num_batches)):
                batch, batch_idx = result
                arr_batch = np.concatenate(batch)
                batch_start = batch_lens[:batch_idx].sum()

                arr[batch_start : batch_start + len(arr_batch)] = arr_batch

                for seq_idx, seq in enumerate(batch):
                    sample_idx = seq_idx + batch_idx * batch_size
                    offsets[sample_idx + 1] = offsets[sample_idx] + len(seq)
                pbar.update(1)
    return arr, offsets


def main(config_path: Path):
    with open(config_path) as f:
        config: Config = Config(**json.load(f))

    os.makedirs(config.out_dir, exist_ok=True)

    datasets = LanguagePairsDatasets(
        data_path=Path(config.datasets_dir),
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        is_test=False,
    )
    rng = np.random.default_rng(seed=config.seed)
    splitter = DataSplitter(
        n_examples=len(datasets), train_frac=config.train_frac, rng=rng
    )
    train_idxs, test_idxs = splitter.split()

    if config.tokenizer_type == "tiktoken":
        enc = tiktoken.get_encoding("gpt2")
    else:
        if config.sentence_piece_model_dir is None:
            raise ValueError("sentence_piece_model_dir required")
        enc = SentencePieceTokenizer(
            model_prefix=str(
                config.sentence_piece_model_dir
                / Path(config.sentence_piece_model_dir).name
            )
        )

    for split_name, idxs in (("train", train_idxs), ("val", test_idxs)):
        num_batches = (len(idxs) + config.batch_size - 1) // config.batch_size

        if config.train_n_tokens is None and config.val_n_tokens is None:
            num_tokens, batch_lens = get_num_tokens_parallel(
                enc=enc,
                datasets=datasets,
                idxs=idxs,
                batch_size=config.batch_size,
                num_batches=num_batches,
                num_workers=config.num_workers,
            )
            np.save(config.out_dir / f"{split_name}_batch_lens.npy", batch_lens)
        else:
            num_tokens = (
                config.train_n_tokens if split_name == "train" else config.val_n_tokens
            )
        print(f"{split_name} num tokens: {num_tokens}")

        arr = np.memmap(
            config.out_dir / f"{split_name}.bin",
            dtype=np.uint16,
            mode="w+",
            shape=(num_tokens,),
        )

        arr, offsets = write_tokens_to_memmap_parallel(
            enc=enc,
            datasets=datasets,
            idxs=idxs,
            batch_size=config.batch_size,
            num_batches=num_batches,
            arr=arr,
            batch_lens=np.load(config.out_dir / f"{split_name}_batch_lens.npy"),
            num_workers=config.num_workers,
            offsets_out_path=config.out_dir / f"{split_name}_offsets.bin",
        )

        arr.flush()
        offsets.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    main(config_path=Path(args.config_path))
