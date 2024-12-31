import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.datasets.wmt14 import WMT14
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)


def train_source_and_target_tokenizers_separately(
    datasets_out_dir: str,
    source_lang: str,
    target_lang: str,
    model_dir: Path,
    source_vocab_size: int = 20000,
    target_vocab_size: int = 20000,
):
    """
    Train 2 tokenizers: a source and a target tokenizer
    :return:
    """
    wmt14 = WMT14(
        out_dir=Path(datasets_out_dir),
        source_lang=source_lang,
        target_lang=target_lang,
        split="train",
    )
    wmt14.download()
    wmt14.write_to_single_file()

    source_model_prefix = Path(model_dir) / f"{source_lang}{source_vocab_size}"
    target_model_prefix = Path(model_dir) / f"{target_lang}{target_vocab_size}"

    print("Constructing SentencePieceTokenizer source model")
    print("=" * 11)
    SentencePieceTokenizer(
        input_path=str(Path(datasets_out_dir) / f"{source_lang}.txt"),
        vocab_size=source_vocab_size,
        model_prefix=str(source_model_prefix),
    )

    print("Constructing SentencePieceTokenizer target model")
    print("=" * 11)
    SentencePieceTokenizer(
        input_path=str(Path(datasets_out_dir) / f"{target_lang}.txt"),
        vocab_size=target_vocab_size,
        model_prefix=str(target_model_prefix),
    )


def main(
    datasets_out_dir: str,
    source_lang: str,
    target_lang: str,
    model_dir: Path,
    source_vocab_size: int = 20000,
    target_vocab_size: int = 20000,
    vocab_size: int = 20000,
    train_single_tokenizer: bool = False
):
    os.makedirs(model_dir, exist_ok=True)

    if train_single_tokenizer:
        model_prefix = Path(model_dir) / f'{vocab_size}'
        SentencePieceTokenizer(
            input_path=[str(Path(datasets_out_dir) / f"{source_lang}.txt"),
                        str(Path(datasets_out_dir) / f"{target_lang}.txt")],
            vocab_size=vocab_size,
            model_prefix=str(model_prefix),
        )
    else:
        train_source_and_target_tokenizers_separately(
            datasets_out_dir=datasets_out_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            model_dir=model_dir
        )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--datasets_out_dir", required=True)
    parser.add_argument("--source_lang", default="fr")
    parser.add_argument("--target_lang", default="en")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--source_vocab_size", default=20000, type=int)
    parser.add_argument("--target_vocab_size", default=20000, type=int)
    parser.add_argument("--vocab_size", default=20000, type=int,
                        help='Vocab size for training a single tokenizer trained on both source and target together')
    parser.add_argument('--train_single_tokenizer', default=False, action='store_true',
                        help='Whether to train a single tokenizer on both source and target together')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        datasets_out_dir=args.datasets_out_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model_dir=args.model_dir,
        source_vocab_size=args.source_vocab_size,
        target_vocab_size=args.target_vocab_size,
        vocab_size=args.vocab_size,
        train_single_tokenizer=args.train_single_tokenizer
    )
