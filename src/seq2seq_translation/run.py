import inspect
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from seq2seq_translation.attention import AttentionType
from seq2seq_translation.data_loading import \
    DataSplitter, CollateFunction, read_data
from seq2seq_translation.sentence_pairs_dataset import SentencePairsDataset
from seq2seq_translation.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.train_evaluate import train, evaluate


def main(
        data_path: str,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        encoder_bidirectional: bool,
        batch_size: int,
        model_weights_out_dir: str,
        n_epochs: int,
        sentence_piece_model_save_dir: str,
        source_tokenizer_train_path: str,
        target_tokenizer_train_path: str,
        limit: Optional[int] = None,
        use_attention: bool = False,
        max_input_length: Optional[int] = None,
        use_pretrained_embeddings: bool = False,
        freeze_embedding_layer: bool = False,
        attention_type: Optional[AttentionType] = None,
        learning_rate: float = 1e-3,
        seed: Optional[int] = None,
        model_weights_path: Optional[str] = None,
        evaluate_only: bool = False,
        min_freq: int = 1,
        source_lang: str = 'en',
        target_lang: str = 'fr',
        source_vocab_length: int = 13000,
        target_vocab_length: int = 13000,
):
    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    if os.environ['USE_WANDB'] == 'True':
        wandb.login()

        signature = inspect.signature(main).parameters.keys()
        wandb.init(
            project="seq2seq_translation",
            config={k: v for k, v in locals().items() if k in signature and k not in (
            'data_path', 'model_weights_out_dir', 'model_weights_path', 'evaluate_only')},
        )

    data = read_data(
        data_path=data_path,
        source_lang=source_lang,
        target_lang=target_lang
    )
    splitter = DataSplitter(
        data=data, train_frac=0.8)
    train_pairs, test_pairs = splitter.split()

    source_tokenizer_model_path = Path(sentence_piece_model_save_dir) / f'{source_lang}{source_vocab_length}'
    target_tokenizer_model_path = Path(sentence_piece_model_save_dir) / f'{target_lang}{source_vocab_length}'

    source_tokenizer = SentencePieceTokenizer(input_path=source_tokenizer_train_path,
                                              vocab_size=source_vocab_length,
                                              model_prefix=str(source_tokenizer_model_path))
    source_tokenizer.train()

    target_tokenizer = SentencePieceTokenizer(input_path=target_tokenizer_train_path,
                                              vocab_size=target_vocab_length,
                                              model_prefix=str(target_tokenizer_model_path))
    target_tokenizer.train()

    print(f'{source_tokenizer.processor.vocab_size()} source tokens')
    print(f'{target_tokenizer.processor.vocab_size()} target tokens')

    if limit is not None:
        train_pairs = train_pairs[:limit]
        test_pairs = test_pairs[:limit]

    train_dset = SentencePairsDataset(
        data=train_pairs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=None,
    )
    val_dset = SentencePairsDataset(
        data=test_pairs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=None,
    )

    collate_fn = CollateFunction(pad_token_id=source_tokenizer.processor.pad_id())
    train_data_loader = DataLoader(
        dataset=train_dset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    val_data_loader = DataLoader(
        dataset=val_dset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(
        input_size=source_tokenizer.processor.vocab_size(),
        hidden_size=encoder_hidden_dim,
        bidirectional=encoder_bidirectional,
        freeze_embedding_layer=freeze_embedding_layer,
        pad_idx=source_tokenizer.processor.pad_id(),
    ).to(device)

    if use_attention:
        decoder = AttnDecoderRNN(
            hidden_size=decoder_hidden_dim,
            attention_size=attention_dim,
            output_size=target_tokenizer.processor.vocab_size(),
            encoder_bidirectional=encoder_bidirectional,
            max_len=max([len(x[1]) for x in train_dset]),
            freeze_embedding_layer=freeze_embedding_layer,
            attention_type=attention_type,
            encoder_output_size=encoder_hidden_dim,
            pad_idx=source_tokenizer.processor.pad_id(),
            num_embeddings=target_tokenizer.processor.vocab_size(),
            sos_token_id=source_tokenizer.processor.bos_id()
        ).to(device)
    else:
        decoder = DecoderRNN(
            hidden_size=decoder_hidden_dim,
            output_size=target_tokenizer.processor.vocab_size(),
            max_len=max([len(x[1]) for x in train_dset]),
            freeze_embedding_layer=freeze_embedding_layer,
            pad_idx=source_tokenizer.processor.pad_id(),
            encoder_hidden_size=2 * encoder_hidden_dim if encoder_bidirectional else
            encoder_hidden_dim,
            num_embeddings=target_tokenizer.processor.vocab_size(),
            sos_token_id=source_tokenizer.processor.bos_id(),
            context_size=2 * encoder_hidden_dim if encoder_bidirectional else encoder_hidden_dim,
        ).to(device)

    if model_weights_path is not None:
        encoder.load_state_dict(
            torch.load(Path(model_weights_path) / 'encoder.pt', map_location=device))
        decoder.load_state_dict(
            torch.load(Path(model_weights_path) / 'decoder.pt', map_location=device))

    if evaluate_only:
        _, val_loss = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_data_loader,
            tokenizer=target_tokenizer,
            criterion=nn.NLLLoss(ignore_index=target_tokenizer.processor.pad_id())
        )
    else:
        train(
            train_dataloader=train_data_loader,
            val_dataloader=val_data_loader,
            encoder=encoder,
            decoder=decoder,
            model_weights_out_dir=model_weights_out_dir,
            n_epochs=n_epochs,
            tokenizer=target_tokenizer
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--encoder_bidirectional', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_weights_out_dir', required=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use_pretrained_embeddings', action='store_true', default=False)
    parser.add_argument('--freeze_embedding_layer', action='store_true', default=False)
    parser.add_argument('--attention_type', default='CosineSimilarityAttention')
    parser.add_argument('--encoder_hidden_dim', default=128, type=int)
    parser.add_argument('--decoder_hidden_dim', default=256, type=int)
    parser.add_argument('--attention_dim', default=256, type=int)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_weights_path', default=None)
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--source_lang', default='fr')
    parser.add_argument('--target_lang', default='en')
    parser.add_argument('--source_tokenizer_train_path', required=True)
    parser.add_argument('--target_tokenizer_train_path', required=True)
    parser.add_argument('--source_vocab_length', default=13000, type=int)
    parser.add_argument('--target_vocab_length', default=13000, type=int)
    parser.add_argument('--sentence_piece_model_save_dir', required=True)

    args = parser.parse_args()

    if not any(args.attention_type == x.value for x in AttentionType):
        raise ValueError(f'Unknown attention type {args.attention_type}')

    os.environ['USE_WANDB'] = str(args.use_wandb)

    main(encoder_bidirectional=args.encoder_bidirectional, batch_size=args.batch_size,
         model_weights_out_dir=args.model_weights_out_dir, n_epochs=args.n_epochs,
         limit=args.limit, use_attention=args.use_attention, data_path=args.data_path,
         max_input_length=args.max_input_length,
         use_pretrained_embeddings=args.use_pretrained_embeddings,
         freeze_embedding_layer=args.freeze_embedding_layer,
         attention_type=AttentionType(args.attention_type),
         encoder_hidden_dim=args.encoder_hidden_dim,
         decoder_hidden_dim=args.decoder_hidden_dim,
         attention_dim=args.attention_dim,
         seed=args.seed,
         model_weights_path=args.model_weights_path,
         evaluate_only=args.evaluate_only,
         min_freq=args.min_freq,
         source_lang=args.source_lang,
         target_lang=args.target_lang,
         source_tokenizer_train_path=args.source_tokenizer_train_path,
         target_tokenizer_train_path=args.target_tokenizer_train_path,
         source_vocab_length=args.source_vocab_length,
         target_vocab_length=args.target_vocab_length,
         sentence_piece_model_save_dir=args.sentence_piece_model_save_dir
         )
