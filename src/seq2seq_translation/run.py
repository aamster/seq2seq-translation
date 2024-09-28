import inspect
import logging
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, DistributedSampler

from seq2seq_translation.attention import AttentionType
from seq2seq_translation.data_loading import \
    DataSplitter, CollateFunction
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.inference import BeamSearchSequenceGenerator, GreedySequenceGenerator
from seq2seq_translation.sentence_pairs_dataset import SentencePairsDataset
from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN, EncoderDecoder
from seq2seq_translation.train_evaluate import train, evaluate
from seq2seq_translation.utils.ddp_utils import DistributedContextManager, \
    SingleProcessContextManager
from torch.nn.parallel import DistributedDataParallel as DDP


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def _remove_module_from_state_dict(state_dict: dict):
    """
    fixing an issue. when training with ddp should have saved model.module.state_dict() instead of model.state_dict()
    removing "module" from keys
    :param state_dict:
    :return:
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v
    return new_state_dict


@record
def main(
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        encoder_bidirectional: bool,
        batch_size: int,
        datasets_dir: str,
        sentence_piece_model_dir: str,
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
        train_frac: float = 0.8,
        git_commit: Optional[str] = None,
        embedding_size: int = 128,
        num_rnn_layers: int = 1,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        compile: bool = False,
        decay_learning_rate: bool = True,
        eval_interval: int = 2000,
        eval_iters: int = 200,
        eval_out_path: Optional[Path] = None,
        is_test: bool = False,
        decoder_num_timesteps: int = 10000,
        use_ddp: bool = False,
        num_train_dataloader_num_workers: int = 0,
        eval_sequence_generator_type: str = 'beam search'
):
    if not evaluate_only and model_weights_out_dir is None:
        raise ValueError('must provide model_weights_out_dir')
    if not evaluate_only and n_epochs is None:
        raise ValueError('must provide n_epochs')

    if use_ddp:
        distributed_context_manager = DistributedContextManager()
    else:
        distributed_context_manager = SingleProcessContextManager()

    with distributed_context_manager as distributed_context:
        if os.environ.get('USE_WANDB') == 'True' and distributed_context.is_master_process:
            wandb.login()

            signature = inspect.signature(main).parameters.keys()

            wandb.init(
                project="seq2seq_translation",
                config={k: v for k, v in locals().items() if k in signature and k not in (
                'data_path', 'model_weights_out_dir', 'model_weights_path', 'evaluate_only')},
            )
            wandb.config.update({"git_commit": git_commit})

        if seed is not None:
            rng = np.random.default_rng(seed=seed)

            # seed_offset used to encourage randomness across processes in ddp
            torch.random.manual_seed(seed + distributed_context.seed_offset)
        else:
            rng = None

        datasets = LanguagePairsDatasets(
            out_dir=Path(datasets_dir),
            source_lang=source_lang,
            target_lang=target_lang,
            is_test=False
        )
        splitter = DataSplitter(
            n_examples=len(datasets), train_frac=train_frac, rng=rng)
        train_idxs, test_idxs = splitter.split()

        source_tokenizer_model_path = Path(sentence_piece_model_dir) / f'{source_lang}'
        target_tokenizer_model_path = Path(sentence_piece_model_dir) / f'{target_lang}'

        source_tokenizer = SentencePieceTokenizer(model_prefix=str(source_tokenizer_model_path))

        target_tokenizer = SentencePieceTokenizer(model_prefix=str(target_tokenizer_model_path))

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
            # TODO would be better if didn't have to truncate
            # consider chunking?
            max_length=max_input_length,
        )
        val_dset = SentencePairsDataset(
            datasets=datasets,
            idxs=test_idxs,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            # TODO would be better if didn't have to truncate
            # consider chunking?
            max_length=max_input_length,
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
            sampler=train_sampler,
            num_workers=num_train_dataloader_num_workers,
            pin_memory=True
        )
        val_data_loader = DataLoader(
            dataset=val_dset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        test_data_loader = DataLoader(
            dataset=test_dset,
            shuffle=False,
            sampler=DistributedSampler(test_dset) if use_ddp else None,
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use_ddp:
            device = f'{device}:{distributed_context.ddp_local_rank}'
        os.environ['DEVICE'] = device

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
            try:
                encoder.load_state_dict(
                    torch.load(Path(model_weights_path) / 'encoder.pt', map_location=device))
                decoder.load_state_dict(
                    torch.load(Path(model_weights_path) / 'decoder.pt', map_location=device))
            except RuntimeError:
                encoder.load_state_dict(
                    _remove_module_from_state_dict(torch.load(Path(model_weights_path) / 'encoder.pt', map_location=device))
                )
                decoder.load_state_dict(
                    _remove_module_from_state_dict(torch.load(Path(model_weights_path) / 'decoder.pt', map_location=device))
                )
        encoder_decoder = EncoderDecoder(
            encoder=encoder,
            decoder=decoder
        )
        if use_ddp:
            encoder_decoder = DDP(encoder_decoder, device_ids=[distributed_context.ddp_local_rank])

        if compile:
            # requires PyTorch 2.0
            print("compiling the model... (takes a ~minute)")
            encoder_decoder = torch.compile(encoder_decoder)

        ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if device.type == 'cuda' else nullcontext()
        logger.info(f'using ctx {ctx}')

        with ctx:
            if evaluate_only:
                val_decoded_text, val_targets, val_bleu, val_bleus, input_lengths = evaluate(
                    encoder_decoder=encoder_decoder,
                    data_loader=test_data_loader if is_test else val_data_loader,
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    sequence_generator_type=BeamSearchSequenceGenerator if eval_sequence_generator_type == 'beam search' else GreedySequenceGenerator
                )
                print(f'bleu: {val_bleu}')
                df = pd.DataFrame(
                    {'bleu': val_bleus,
                     'input_length': input_lengths,
                     'pred': val_decoded_text,
                     'target': val_targets
                     })
                df.to_csv(eval_out_path.parent / f'{eval_out_path.stem}_{distributed_context.ddp_local_rank}.csv', index=False)
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
    parser.add_argument('--max_input_length', type=int, default=300)
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
    parser.add_argument('--sentence_piece_model_dir', required=True)
    parser.add_argument('--train_frac', type=float, default=0.8)
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
    parser.add_argument('--num_train_dataloader_workers', type=int, default=0)
    parser.add_argument('--eval_sequence_generator_type', default='beam search')
    args = parser.parse_args()

    if not any(args.attention_type == x.value for x in AttentionType):
        raise ValueError(f'Unknown attention type {args.attention_type}')

    os.environ['USE_WANDB'] = str(args.use_wandb)
    if args.use_wandb:
        if args.wandb_api_key is None:
            raise ValueError('Must provide wandb_api_key')
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
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
         sentence_piece_model_dir=args.sentence_piece_model_dir,
         datasets_dir=args.datasets_dir,
         train_frac=args.train_frac,
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
         eval_out_path=Path(args.eval_out_path),
         is_test=args.is_test,
         decoder_num_timesteps=args.decoder_num_timesteps,
         use_ddp=args.use_ddp,
         num_train_dataloader_num_workers=args.num_train_dataloader_workers,
         eval_sequence_generator_type=args.eval_sequence_generator_type
         )
