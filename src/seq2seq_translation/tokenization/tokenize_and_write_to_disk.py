import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict

from seq2seq_translation.data_loading import DataSplitter
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets

@dataclass
class Config:
    datasets_dir: Path
    source_lang: str
    target_lang: str
    seed: int
    out_dir: Path
    train_frac: float = 0.999
    tokenizer_type: str = 'tiktoken'
    max_len: int = 128

    def __post_init__(self):
        self.datasets_dir = Path(self.datasets_dir)
        self.out_dir = Path(self.out_dir)

def main(config_path: Path):
    with open(config_path) as f:
        config: Config = Config(**json.load(f))

    os.makedirs(config.out_dir, exist_ok=True)

    datasets = LanguagePairsDatasets(
        out_dir=Path(config.datasets_dir),
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        is_test=False,
    )
    rng = np.random.default_rng(seed=config.seed)
    splitter = DataSplitter(
        n_examples=len(datasets), train_frac=config.train_frac, rng=rng
    )
    train_idxs, test_idxs = splitter.split()

    if config.tokenizer_type == 'tiktoken':
        enc = tiktoken.get_encoding("gpt2")
    else:
        raise NotImplemented

    def process(example):
        source = example['source']
        target = example['target']

        source_ids = enc.encode_ordinary(source)
        target_ids = enc.encode_ordinary(target)

        source_ids = source_ids[:config.max_len]
        target_ids = target_ids[:config.max_len]

        source_ids.append(enc.eot_token)
        target_ids.append(enc.eot_token)

        combined = source_ids + target_ids

        out = {'ids': combined, 'len': len(combined)}
        return out


    def dataset_gen(idxs: np.ndarray):
        for idx in idxs:
            source, target, dataset_name = datasets[idx]
            yield {'source': source, 'target': target}

    train = Dataset.from_generator(lambda: dataset_gen(idxs=train_idxs), num_proc=os.cpu_count() // 2)
    val = Dataset.from_generator(lambda: dataset_gen(idxs=test_idxs), num_proc=os.cpu_count() // 2)

    split_dataset = DatasetDict({'train': train, 'val': val})

    tokenized = split_dataset.map(
        process,
        desc="tokenizing the splits",
        num_proc=os.cpu_count() // 2,
    )

    for split, dset in tokenized.items():
        dset.set_format("numpy")

        total_tokens = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split}: total tokens = {total_tokens}")

        dtype = np.uint16  # Safe since the GPT2 vocab size (50257 incl. EOT) < 65536
        arr = np.memmap(
            filename=config.out_dir / f"{split}.bin",
            dtype=dtype,
            mode="w+",
            shape=(total_tokens,),
        )

        # Prepare offsets array for each sample. offsets[i] = starting index of sample i in arr
        # => offsets[len(dset)] will be total_tokens.
        offsets = np.zeros(len(dset) + 1, dtype=np.uint64)
        offsets[0] = 0

        total_shards = min(1024, 4)
        idx = 0            # Current position in arr (token-level)
        sample_idx = 0     # Current position in offsets (sample-level)

        for shard_idx in tqdm(range(total_shards), desc=f"Writing {split}"):
            shard = dset.shard(num_shards=total_shards, index=shard_idx, contiguous=True)

            shard_ids = shard["ids"]  # This is a list of lists/arrays of token IDs
            arr_batch = np.concatenate(shard_ids)

            # Write the tokens in this shard to the memmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

            # Update offsets for each sample in this shard
            for seq in shard_ids:
                offsets[sample_idx + 1] = offsets[sample_idx] + len(seq)
                sample_idx += 1

        # Make sure everything is written to disk
        arr.flush()

        # Save the offsets array
        np.save(config.out_dir / f"{split}_offsets.npy", offsets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()
    main(config_path=Path(args.config_path))
