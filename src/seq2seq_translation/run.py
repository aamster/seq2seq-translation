import json
import os
from argparse import ArgumentParser
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from seq2seq_translation.config._config import ModelType, TokenizerType
from seq2seq_translation.config.rnn_config import RNNConfig
from seq2seq_translation.config.transformer_config import TransformerConfig, GPT2Size
from seq2seq_translation.data_loading import CollateFunction
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.inference import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from seq2seq_translation.models.transformer.decoder import DecoderTransformer
from seq2seq_translation.models.transformer.encoder_decoder import (
    EncoderDecoderTransformer,
)
from seq2seq_translation.models.transformer.mlp import ActivationFunction
from seq2seq_translation.models.transformer.positional_encoding import (
    PositionalEncodingType,
)
from seq2seq_translation.sentence_pairs_dataset import (
    SentencePairsDataset,
    SentencePairsDatasetFromPreprocessedTokens,
)
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)
from seq2seq_translation.models.rnn import (
    EncoderRNN,
    DecoderRNN,
    AttnDecoderRNN,
    EncoderDecoderRNN,
)
from seq2seq_translation.tokenization.tiktoken_tokenizer import TikTokenTokenizer
from seq2seq_translation.train_evaluate import train, evaluate
from seq2seq_translation.utils.ddp_utils import (
    DistributedContextManager,
    SingleProcessContextManager,
)
from torch.nn.parallel import DistributedDataParallel as DDP


def _fix_model_state_dict(state_dict: dict):
    """
    :param state_dict:
    :return:
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.` prefix
        name = name.replace("_orig_mod.", "")

        name = name.replace(
            "positional_encoding", "positional_embedding"
        )  # seems this has been renamed
        new_state_dict[name] = v
    return new_state_dict


def main(config: RNNConfig | TransformerConfig):
    if isinstance(config, TransformerConfig) and config.from_gpt2_weights:
        if config.gpt_size is None:
            raise ValueError("specify gpt2 size")
        if config.gpt_size == GPT2Size.SMALL:
            config.num_layers = 12
            config.n_head = 12
            config.d_model = 768
        elif config.gpt_size == GPT2Size.MEDIUM:
            config.num_layers = 24
            config.n_head = 16
            config.d_model = 1024
        elif config.gpt_size == GPT2Size.LARGE:
            config.num_layers = 36
            config.n_head = 20
            config.d_model = 1280
        config.feedforward_hidden_dim = config.d_model * 4
        config.norm_first = True
        config.activation = ActivationFunction.GELU
        config.positional_encoding_type = PositionalEncodingType.LEARNED
        config.fixed_length = 1024
        config.decoder_only = True

    if not config.evaluate_only and config.weights_out_dir is None:
        raise ValueError("must provide model_weights_out_dir")
    if not config.evaluate_only and config.n_epochs is None:
        raise ValueError("must provide n_epochs")

    if config.use_ddp:
        distributed_context_manager = DistributedContextManager()
    else:
        distributed_context_manager = SingleProcessContextManager()

    with distributed_context_manager as distributed_context:
        if (
            os.environ.get("USE_WANDB") == "True"
            and distributed_context.is_master_process
        ):
            wandb.login()

            wandb.init(
                project="seq2seq_translation",
                config={
                    k: v
                    for k, v in config.model_dump().items()
                    if k
                    not in (
                        "data_path",
                        "model_weights_out_dir",
                        "model_weights_path",
                        "evaluate_only",
                    )
                },
            )
            wandb.config.update({"git_commit": config.git_commit})

        if config.seed is not None:
            rng = np.random.default_rng(seed=config.seed)

            # seed_offset used to encourage randomness across processes in ddp
            torch.random.manual_seed(config.seed + distributed_context.seed_offset)
        else:
            rng = None

        if not config.is_test:
            train_offsets = np.memmap(
                config.tokenized_dir / "train_offsets.bin", dtype=np.uint64
            )
            train_tokenized = np.memmap(
                config.tokenized_dir / "train.bin", dtype=np.uint16
            )

            val_offsets = np.memmap(
                config.tokenized_dir / "val_offsets.bin", dtype=np.uint64
            )
            val_tokenized = np.memmap(config.tokenized_dir / "val.bin", dtype=np.uint16)

            # -1 because it goes until 1 past the last sequence
            train_idxs = np.arange(len(train_offsets) - 1)
            rng.shuffle(train_idxs)

            # -1 because it goes until 1 past the last sequence
            test_idxs = np.arange(len(val_offsets) - 1)
            rng.shuffle(test_idxs)

        if config.decoder_only:
            if config.tokenizer_type == TokenizerType.SENTENCEPIECE:
                tokenizer = SentencePieceTokenizer(
                    model_prefix=str(
                        Path(config.sentence_piece_model_dir)
                        / Path(config.sentence_piece_model_dir).name
                    ),
                    include_language_tag=config.include_language_tag,
                )
                eot_token_id = tokenizer.processor.eos_id()
                logger.info(f"{tokenizer.processor.vocab_size()} tokens")
                source_tokenizer = tokenizer
                target_tokenizer = None
            else:
                tokenizer = TikTokenTokenizer()
                eot_token_id = tokenizer.tokenizer.eot_token
                logger.info(f"{tokenizer.vocab_size} vocab size")

        else:
            if config.tokenizer_type == TokenizerType.SENTENCEPIECE:
                if config.use_separate_tokenizer_for_source_target_lang:
                    source_tokenizer = SentencePieceTokenizer(
                        model_prefix=str(
                            Path(config.sentence_piece_model_dir) / config.source_lang
                        ),
                        include_language_tag=config.include_language_tag,
                    )
                    target_tokenizer = SentencePieceTokenizer(
                        model_prefix=str(
                            Path(config.sentence_piece_model_dir) / config.target_lang
                        ),
                        include_language_tag=config.include_language_tag,
                    )
                    logger.info(
                        f"source {source_tokenizer.processor.vocab_size()} vocab size"
                    )
                    logger.info(
                        f"target {target_tokenizer.processor.vocab_size()} vocab size"
                    )
                    eot_token_id = source_tokenizer.processor.eos_id()
                    tokenizer = None
                else:
                    tokenizer = SentencePieceTokenizer(
                        model_prefix=str(
                            Path(config.sentence_piece_model_dir)
                            / Path(config.sentence_piece_model_dir).name
                        ),
                        include_language_tag=config.include_language_tag,
                    )
                    source_tokenizer = None
                    target_tokenizer = None
                    logger.info(f"{tokenizer.processor.vocab_size()} vocab size")
                    eot_token_id = tokenizer.processor.eos_id()

            else:
                tokenizer = TikTokenTokenizer()
                eot_token_id = tokenizer.tokenizer.eot_token
                logger.info(f"{tokenizer.vocab_size} vocab size")

        if config.limit is not None:
            train_idxs = train_idxs[: config.limit]
            test_idxs = test_idxs[: config.limit]
            print(f"Number of train examples after limiting: {len(train_idxs)}")
            print(f"Number of val examples after limiting: {len(test_idxs)}")

        if config.is_test:
            test_datasets = LanguagePairsDatasets(
                out_dir=Path(config.dataset_path),
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                is_test=True,
            )

            test_dset = SentencePairsDataset(
                datasets=test_datasets,
                idxs=np.arange(len(test_datasets)),
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                combined_tokenizer=tokenizer,
                combine_source_and_target=config.decoder_only,
                max_length=None,
                eos_token_id=eot_token_id,
                pad_token_id=(
                    source_tokenizer.pad_idx
                    if config.use_separate_tokenizer_for_source_target_lang
                    else tokenizer.pad_idx
                ),
                source_language_tag_token_id=(
                    tokenizer.language_tag_map[config.source_lang]
                    if config.include_language_tag
                    else None
                ),
                target_language_tag_token_id=(
                    tokenizer.language_tag_map[config.target_lang]
                    if config.include_language_tag
                    else None
                ),
            )
            test_data_loader = DataLoader(
                dataset=test_dset,
                shuffle=False,
                sampler=DistributedSampler(test_dset) if config.use_ddp else None,
                batch_size=config.batch_size,
                collate_fn=CollateFunction(
                    pad_token_id=(
                        source_tokenizer.pad_idx
                        if config.use_separate_tokenizer_for_source_target_lang
                        else tokenizer.pad_idx
                    )
                ),
            )

        else:
            train_dset = SentencePairsDatasetFromPreprocessedTokens(
                idxs=train_idxs,
                combine_source_and_target=config.decoder_only,
                tokenized_offsets=train_offsets,
                tokenized=train_tokenized,
                eot_token_id=eot_token_id,
                pad_token_id=tokenizer.pad_idx,
                source_language_tag_token_id=tokenizer.language_tag_map[
                    config.source_lang
                ],
                target_language_tag_token_id=tokenizer.language_tag_map[
                    config.target_lang
                ],
            )
            val_dset = SentencePairsDatasetFromPreprocessedTokens(
                idxs=test_idxs,
                combine_source_and_target=config.decoder_only,
                tokenized_offsets=val_offsets,
                tokenized=val_tokenized,
                eot_token_id=eot_token_id,
                pad_token_id=tokenizer.pad_idx,
                source_language_tag_token_id=tokenizer.language_tag_map[
                    config.source_lang
                ],
                target_language_tag_token_id=tokenizer.language_tag_map[
                    config.target_lang
                ],
            )

            train_sampler = DistributedSampler(train_dset) if config.use_ddp else None
            train_data_loader = DataLoader(
                dataset=train_dset,
                shuffle=(train_sampler is None),
                batch_size=config.batch_size,
                collate_fn=CollateFunction(
                    pad_token_id=tokenizer.pad_idx, fixed_length=config.fixed_length
                ),
                sampler=train_sampler,
                num_workers=config.num_train_dataloader_num_workers,
                pin_memory=True,
            )
            val_data_loader = DataLoader(
                dataset=val_dset,
                shuffle=False,
                batch_size=config.batch_size,
                collate_fn=CollateFunction(pad_token_id=tokenizer.pad_idx),
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if config.use_ddp:
            device = f"{device}:{distributed_context.ddp_local_rank}"
        os.environ["DEVICE"] = device

        device = torch.device(device)

        if config.architecture_type == ModelType.RNN:
            assert isinstance(config, RNNConfig), "expected RNNConfig"
            encoder = EncoderRNN(
                input_size=tokenizer.processor.vocab_size(),
                hidden_size=config.encoder_hidden_dim,
                bidirectional=config.encoder_bidirectional,
                freeze_embedding_layer=config.freeze_embedding_layer,
                pad_idx=tokenizer.processor.pad_id(),
                embedding_dim=config.embedding_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ).to(device)

            if config.use_attention:
                decoder = AttnDecoderRNN(
                    hidden_size=config.decoder_hidden_dim,
                    attention_size=config.attention_dim,
                    output_size=tokenizer.processor.vocab_size(),
                    encoder_bidirectional=config.encoder_bidirectional,
                    max_len=config.decoder_num_timesteps,
                    freeze_embedding_layer=config.freeze_embedding_layer,
                    attention_type=config.attention_type,
                    encoder_output_size=encoder.output_size,
                    pad_idx=tokenizer.processor.pad_id(),
                    num_embeddings=tokenizer.processor.vocab_size(),
                    sos_token_id=tokenizer.processor.bos_id(),
                    embedding_dim=config.embedding_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    eos_token_id=tokenizer.processor.eos_id(),
                ).to(device)
            else:
                decoder = DecoderRNN(
                    hidden_size=config.decoder_hidden_dim,
                    output_size=tokenizer.processor.vocab_size(),
                    max_len=config.decoder_num_timesteps,
                    freeze_embedding_layer=config.freeze_embedding_layer,
                    pad_idx=tokenizer.processor.pad_id(),
                    encoder_output_size=encoder.output_size,
                    num_embeddings=tokenizer.processor.vocab_size(),
                    sos_token_id=tokenizer.processor.bos_id(),
                    context_size=int(encoder.hidden_size / config.num_layers),
                    embedding_dim=config.embedding_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    encoder_bidirectional=config.encoder_bidirectional,
                    eos_token_id=tokenizer.processor.eos_id(),
                ).to(device)

            model = EncoderDecoderRNN(encoder=encoder, decoder=decoder)
        else:
            assert isinstance(config, TransformerConfig), "expected TransformerConfig"
            if config.decoder_only:
                if config.from_gpt2_weights:
                    model = DecoderTransformer.from_pretrained(
                        config=config,
                        vocab_size=tokenizer.vocab_size,
                        model_type="gpt2",
                        override_args=dict(dropout=config.dropout),
                        pad_token_idx=tokenizer.pad_idx,
                    ).to(device)
                else:
                    model = DecoderTransformer(
                        n_attention_heads=config.n_head,
                        n_layers=config.num_layers,
                        vocab_size=tokenizer.vocab_size,
                        d_model=config.d_model,
                        block_size=config.fixed_length,
                        feedforward_hidden_dim=config.feedforward_hidden_dim,
                        norm_first=config.norm_first,
                        mlp_activation=config.activation,
                        use_cross_attention=False,
                        positional_encoding_type=config.positional_encoding_type,
                        pad_token_idx=tokenizer.pad_idx,
                    ).to(device)
            else:
                model = EncoderDecoderTransformer(
                    n_attention_heads=config.n_head,
                    n_layers=config.num_layers,
                    vocab_size=(
                        source_tokenizer.vocab_size
                        if config.use_separate_tokenizer_for_source_target_lang
                        else tokenizer.vocab_size
                    ),
                    d_model=config.d_model,
                    block_size=config.max_input_length,
                    feedforward_hidden_dim=config.feedforward_hidden_dim,
                    sos_token_id=(
                        source_tokenizer.processor.bos_id()
                        if config.use_separate_tokenizer_for_source_target_lang
                        else tokenizer.processor.bos_id()
                    ),
                    eos_token_id=(
                        source_tokenizer.processor.eos_id()
                        if config.use_separate_tokenizer_for_source_target_lang
                        else tokenizer.processor.eos_id()
                    ),
                    pad_token_id=(
                        source_tokenizer.processor.pad_id()
                        if config.use_separate_tokenizer_for_source_target_lang
                        else tokenizer.processor.pad_id()
                    ),
                    norm_first=config.norm_first,
                    mlp_activation=config.activation,
                    positional_encoding_type=config.positional_encoding_type,
                ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if config.load_from_checkpoint_path is not None:
            checkpoint = torch.load(
                config.load_from_checkpoint_path, map_location=device
            )
            try:
                model.load_state_dict(checkpoint["model"])
            except RuntimeError:
                model.load_state_dict(_fix_model_state_dict(checkpoint["model"]))

            optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info(f"Model num params: {model.num_params / 1e6}M")

        if config.compile:
            logger.info("compiling model")
            model = torch.compile(model)

        if config.use_ddp:
            model = DDP(model, device_ids=[distributed_context.ddp_local_rank])

        if device.type == "cuda" and config.use_mixed_precision:
            if config.dtype == "bfloat16":
                model_dtype = torch.bfloat16
            elif config.dtype == "float16":
                model_dtype = torch.float16

            ctx = torch.amp.autocast(device.type, dtype=model_dtype)
        else:
            ctx = nullcontext()
        logger.info(f"using ctx {ctx}")

        if config.evaluate_only:
            with ctx:
                val_decoded_text, val_targets, val_bleu, val_bleus, input_lengths = (
                    evaluate(
                        model=model,
                        data_loader=(
                            test_data_loader if config.is_test else val_data_loader
                        ),
                        tokenizer=(
                            target_tokenizer
                            if config.use_separate_tokenizer_for_source_target_lang
                            else tokenizer
                        ),
                        source_tokenizer=source_tokenizer,
                        sequence_generator_type=(
                            BeamSearchSequenceGenerator
                            if config.eval_sequence_generator_type == "beam search"
                            else GreedySequenceGenerator
                        ),
                    )
                )
            print(f"bleu: {val_bleu}")
            df = pd.DataFrame(
                {
                    "bleu": val_bleus,
                    "input_length": input_lengths,
                    "pred": val_decoded_text,
                    "target": val_targets,
                }
            )
            df.to_csv(
                config.eval_out_path.parent
                / f"{config.eval_out_path.stem}_{distributed_context.ddp_local_rank}.csv",
                index=False,
            )
        else:
            train(
                train_dataloader=train_data_loader,
                val_dataloader=val_data_loader,
                model=model,
                optimizer=optimizer,
                model_weights_out_dir=str(config.weights_out_dir),
                n_epochs=config.n_epochs,
                tokenizer=tokenizer,
                learning_rate=config.learning_rate,
                decay_learning_rate=config.decay_learning_rate,
                loss_eval_interval=config.loss_eval_interval,
                accuracy_eval_interval=config.accuracy_eval_interval,
                eval_iters=config.eval_iters,
                label_smoothing=config.label_smoothing,
                autocast_context=ctx,
                max_new_inference_tokens=config.decoder_num_timesteps,
                loss_type=config.loss_type,
            )


def _record(main_func, config):
    if config.use_ddp:
        from torch.distributed.elastic.multiprocessing.errors import record

        return record(main_func)
    return main_func


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    if config["architecture_type"] == ModelType.RNN.value:
        config = RNNConfig.model_validate(config)
    elif config["architecture_type"] == ModelType.TRANSFORMER.value:
        config = TransformerConfig.model_validate(config)
    else:
        raise ValueError(f'unknown architecture_type {config["architecture_type"]}')

    os.environ["USE_WANDB"] = str(config.use_wandb)
    if config.use_wandb:
        if config.wandb_api_key is None:
            raise ValueError("Must provide wandb_api_key")
        os.environ["WANDB_API_KEY"] = config.wandb_api_key

    main = _record(main_func=main, config=config)
    main(config=config)