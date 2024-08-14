import inspect
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from seq2seq_translation.attention import AttentionType
from seq2seq_translation.data_loading import \
    DataSplitter, CollateFunction
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.sentence_pairs_dataset import SentencePairsDataset
from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.train_evaluate import train, evaluate
from seq2seq_translation.utils.ddp_utils import init_ddp


def main(
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        encoder_bidirectional: bool,
        batch_size: int,
        datasets_dir: str,
        sentence_piece_model_dir: str,
        source_tokenizer_train_path: str,
        target_tokenizer_train_path: str,
        n_epochs: Optional[int] = None,
        model_weights_out_dir: Optional[str] = None,
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
        train_frac: float = 0.8,
        dataset_sample_fracs: Optional[list[float]] = None,
        git_commit: Optional[str] = None,
        embedding_size: int = 128,
        num_rnn_layers: int = 1,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        compile: bool = False,
        decay_learning_rate: bool = True,
        eval_interval: int = 2000,
        eval_iters: int = 200,
        eval_out_path: Optional[str] = None,
        is_test: bool = False,
        decoder_num_timesteps: int = 10000,
        use_ddp: bool = False
):
    if not evaluate_only and model_weights_out_dir is None:
        raise ValueError('must provide model_weights_out_dir')
    if not evaluate_only and n_epochs is None:
        raise ValueError('must provide n_epochs')

    if os.environ.get('USE_WANDB') == 'True':
        wandb.login()

        signature = inspect.signature(main).parameters.keys()
        wandb_run = wandb.init(
            project="seq2seq_translation",
            config={k: v for k, v in locals().items() if k in signature and k not in (
            'data_path', 'model_weights_out_dir', 'model_weights_path', 'evaluate_only')},
        )
        wandb.config.update({"git_commit": git_commit})

    if use_ddp:
        ddp_world_size, master_process, seed_offset, ddp_local_rank = init_ddp()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = None

    if seed is not None:
        rng = np.random.default_rng(1234)

        # seed_offset used to encourage randomness across processes in ddp
        torch.random.manual_seed(seed + seed_offset)
    else:
        rng = None

    datasets = LanguagePairsDatasets(
        out_dir=Path(datasets_dir),
        source_lang=source_lang,
        target_lang=target_lang,
        is_test=False
    )
    print('Creating source tokenizer train set')
    datasets.create_source_tokenizer_train_set(
        source_tokenizer_path=Path(source_tokenizer_train_path)
    )
    print('Creating target tokenizer train set')
    datasets.create_target_tokenizer_train_set(
        target_tokenizer_path=Path(target_tokenizer_train_path)
    )

    splitter = DataSplitter(
        n_examples=len(datasets), train_frac=train_frac, rng=rng)
    train_idxs, test_idxs = splitter.split()

    source_tokenizer_model_path = Path(sentence_piece_model_dir) / f'{source_lang}'
    target_tokenizer_model_path = Path(sentence_piece_model_dir) / f'{target_lang}'

    source_tokenizer = SentencePieceTokenizer(input_path=source_tokenizer_train_path,
                                              vocab_size=source_vocab_length,
                                              model_prefix=str(source_tokenizer_model_path))

    target_tokenizer = SentencePieceTokenizer(input_path=target_tokenizer_train_path,
                                              vocab_size=target_vocab_length,
                                              model_prefix=str(target_tokenizer_model_path))

    print(f'{source_tokenizer.processor.vocab_size()} source tokens')
    print(f'{target_tokenizer.processor.vocab_size()} target tokens')

    if limit is not None:
        train_idxs = train_idxs[:int(len(train_idxs) * limit)]
        test_idxs = test_idxs[:int(len(test_idxs) * limit)]
        print(f'Number of train examples after limiting: {len(train_idxs)}')
        print(f'Number of val examples after limiting: {len(test_idxs)}')

    train_dset = SentencePairsDataset(
        datasets=datasets,
        idxs=train_idxs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=max_input_length,
    )
    val_dset = SentencePairsDataset(
        datasets=datasets,
        idxs=test_idxs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=None,
    )

    test_datasets = LanguagePairsDatasets(
            out_dir=Path(datasets_dir),
            source_lang=source_lang,
            target_lang=target_lang,
            is_test=True
    )

    test_dset = SentencePairsDataset(
        datasets=test_datasets,
        idxs=np.arange(len(test_datasets)),
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=None,
    )

    collate_fn = CollateFunction(pad_token_id=source_tokenizer.processor.pad_id())

    train_sampler = DistributedSampler(train_dset) if use_ddp else None
    train_data_loader = DataLoader(
        dataset=train_dset,
        shuffle=(train_sampler is None),
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    val_data_loader = DataLoader(
        dataset=val_dset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        dataset=TensorDataset(test_dset[0][0].unsqueeze(0), test_dset[0][1].unsqueeze(0), torch.tensor([]).unsqueeze(0)),
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    if torch.cuda.is_available():
        if use_ddp:
            device = f"cuda:{ddp_local_rank}"
        else:
            device = 'cuda'
    else:
        if use_ddp:
            raise ValueError('Cannot use ddp on cpu')
        else:
            device = 'cpu'
    device = torch.device(device)

    encoder = EncoderRNN(
        input_size=source_tokenizer.processor.vocab_size(),
        hidden_size=encoder_hidden_dim,
        bidirectional=encoder_bidirectional,
        freeze_embedding_layer=freeze_embedding_layer,
        pad_idx=source_tokenizer.processor.pad_id(),
        embedding_dim=embedding_size,
        num_layers=num_rnn_layers,
        dropout=dropout,
    ).to(device)

    if use_attention:
        decoder = AttnDecoderRNN(
            hidden_size=decoder_hidden_dim,
            attention_size=attention_dim,
            output_size=target_tokenizer.processor.vocab_size(),
            encoder_bidirectional=encoder_bidirectional,
            max_len=decoder_num_timesteps,
            freeze_embedding_layer=freeze_embedding_layer,
            attention_type=attention_type,
            encoder_output_size=encoder.output_size,
            pad_idx=source_tokenizer.processor.pad_id(),
            num_embeddings=target_tokenizer.processor.vocab_size(),
            sos_token_id=source_tokenizer.processor.bos_id(),
            embedding_dim=embedding_size,
            num_layers=num_rnn_layers,
            dropout=dropout,
            eos_token_id=target_tokenizer.processor.eos_id()
        ).to(device)
    else:
        decoder = DecoderRNN(
            hidden_size=decoder_hidden_dim,
            output_size=target_tokenizer.processor.vocab_size(),
            max_len=decoder_num_timesteps,
            freeze_embedding_layer=freeze_embedding_layer,
            pad_idx=source_tokenizer.processor.pad_id(),
            encoder_output_size=encoder.output_size,
            num_embeddings=target_tokenizer.processor.vocab_size(),
            sos_token_id=source_tokenizer.processor.bos_id(),
            context_size=int(encoder.hidden_size / num_rnn_layers),
            embedding_dim=embedding_size,
            num_layers=num_rnn_layers,
            dropout=dropout,
            encoder_bidirectional=encoder_bidirectional,
            eos_token_id=target_tokenizer.processor.eos_id()
        ).to(device)

    if model_weights_path is not None:
        encoder.load_state_dict(
            torch.load(Path(model_weights_path) / 'encoder.pt', map_location=device))
        decoder.load_state_dict(
            torch.load(Path(model_weights_path) / 'decoder.pt', map_location=device))

    if compile:
        # requires PyTorch 2.0
        print("compiling the model... (takes a ~minute)")
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)

    if evaluate_only:
        val_decoded_text, val_targets, val_bleu, val_bleus, input_lengths = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=test_data_loader if is_test else val_data_loader,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        print(f'bleu: {val_bleu}')
        df = pd.DataFrame(
            {'bleu': val_bleus,
             'input_length': input_lengths,
             'pred': val_decoded_text,
             'target': val_targets
             })
        df.to_csv(eval_out_path, index=False)
    else:
        train(
            train_dataloader=train_data_loader,
            val_dataloader=val_data_loader,
            encoder=encoder,
            decoder=decoder,
            model_weights_out_dir=model_weights_out_dir,
            n_epochs=n_epochs,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            decay_learning_rate=decay_learning_rate,
            eval_interval=eval_interval,
            eval_iters=eval_iters
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--encoder_bidirectional', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_weights_out_dir')
    parser.add_argument('--datasets_dir', required=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--limit', type=float, default=None)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--use_pretrained_embeddings', action='store_true', default=False)
    parser.add_argument('--freeze_embedding_layer', action='store_true', default=False)
    parser.add_argument('--attention_type', default='CosineSimilarityAttention')
    parser.add_argument('--encoder_hidden_dim', default=128, type=int)
    parser.add_argument('--decoder_hidden_dim', default=256, type=int)
    parser.add_argument('--attention_dim', default=256, type=int)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_api_key', default=None)
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
    parser.add_argument('--sentence_piece_model_dir', required=True)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--dataset_sample_fracs', default=None, help='amount to sample for each dataset. Should be of form "0.7 1.0"')
    parser.add_argument('--git_commit', default=None)
    parser.add_argument('--num_rnn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--compile', default=False, action='store_true')
    parser.add_argument('--decay_learning_rate', default=False, action='store_true')
    parser.add_argument('--eval_interval', default=2000, type=int, help='How often to evaluate performance')
    parser.add_argument('--eval_iters', default=200, type=int, help='How many batches of data to use for evaluation')
    parser.add_argument('--eval_out_path', help='Where to save eval metrics')
    parser.add_argument('--is_test', action='store_true', default=False)
    parser.add_argument('--decoder_num_timesteps', type=int, default=10000)
    parser.add_argument('--use_ddp', action='store_true', default=False)
    args = parser.parse_args()

    if not any(args.attention_type == x.value for x in AttentionType):
        raise ValueError(f'Unknown attention type {args.attention_type}')

    os.environ['USE_WANDB'] = str(args.use_wandb)
    if args.use_wandb:
        if args.wandb_api_key is None:
            raise ValueError('Must provide wandb_api_key')
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
    if args.dataset_sample_fracs:
        dataset_sample_fracs = [float(x) for x in args.dataset_sample_fracs.split(' ')]
    else:
        dataset_sample_fracs = None
    main(encoder_bidirectional=args.encoder_bidirectional, batch_size=args.batch_size,
         model_weights_out_dir=args.model_weights_out_dir, n_epochs=args.n_epochs,
         limit=args.limit, use_attention=args.use_attention,
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
         sentence_piece_model_dir=args.sentence_piece_model_dir,
         datasets_dir=args.datasets_dir,
         train_frac=args.train_frac,
         dataset_sample_fracs=dataset_sample_fracs,
         git_commit=args.git_commit,
         embedding_size=args.embedding_dim,
         num_rnn_layers=args.num_rnn_layers,
         learning_rate=args.learning_rate,
         dropout=args.dropout,
         weight_decay=args.weight_decay,
         decay_learning_rate=args.decay_learning_rate,
         compile=args.compile,
         eval_interval=args.eval_interval,
         eval_iters=args.eval_iters,
         eval_out_path=args.eval_out_path,
         is_test=args.is_test,
         decoder_num_timesteps=args.decoder_num_timesteps
         )
