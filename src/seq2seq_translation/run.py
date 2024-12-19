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
from torch.nn import Transformer
from torch.utils.data import DataLoader, DistributedSampler

from seq2seq_translation.config._config import ModelType
from seq2seq_translation.config.rnn_config import RNNConfig
from seq2seq_translation.config.transformer_config import TransformerConfig
from seq2seq_translation.data_loading import DataSplitter, CollateFunction
from seq2seq_translation.datasets.datasets import LanguagePairsDatasets
from seq2seq_translation.inference import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from seq2seq_translation.models.transformer.encoder_decoder import EncoderDecoderTransformer2
from seq2seq_translation.sentence_pairs_dataset import SentencePairsDataset
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)
from seq2seq_translation.models.rnn import (
    EncoderRNN,
    DecoderRNN,
    AttnDecoderRNN,
    EncoderDecoderRNN,
)
from seq2seq_translation.train_evaluate import train, evaluate
from seq2seq_translation.utils.ddp_utils import (
    DistributedContextManager,
    SingleProcessContextManager,
)
from torch.nn.parallel import DistributedDataParallel as DDP


def _remove_module_from_state_dict(state_dict: dict):
    """
    fixing an issue. when training with ddp should have saved model.module.state_dict() instead of model.state_dict()
    removing "module" from keys
    :param state_dict:
    :return:
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.` prefix
        new_state_dict[name] = v
    return new_state_dict


def main(config: RNNConfig | TransformerConfig):
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

        datasets = LanguagePairsDatasets(
            out_dir=Path(config.datasets_dir),
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            is_test=False,
        )
        splitter = DataSplitter(
            n_examples=len(datasets), train_frac=config.train_frac, rng=rng
        )
        train_idxs, test_idxs = splitter.split()

        source_tokenizer_model_path = (
            Path(config.sentence_piece_model_dir) / f"{config.source_lang}"
        )
        target_tokenizer_model_path = (
            Path(config.sentence_piece_model_dir) / f"{config.target_lang}"
        )

        source_tokenizer = SentencePieceTokenizer(
            model_prefix=str(source_tokenizer_model_path)
        )

        target_tokenizer = SentencePieceTokenizer(
            model_prefix=str(target_tokenizer_model_path)
        )

        print(f"{source_tokenizer.processor.vocab_size()} source tokens")
        print(f"{target_tokenizer.processor.vocab_size()} target tokens")

        if config.limit is not None:
            train_idxs = train_idxs[: config.limit]
            test_idxs = test_idxs[: config.limit]
            print(f"Number of train examples after limiting: {len(train_idxs)}")
            print(f"Number of val examples after limiting: {len(test_idxs)}")

        train_dset = SentencePairsDataset(
            datasets=datasets,
            idxs=train_idxs,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            # TODO would be better if didn't have to truncate
            # consider chunking?
            max_length=config.max_input_length,
        )
        val_dset = SentencePairsDataset(
            datasets=datasets,
            idxs=test_idxs,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            # TODO would be better if didn't have to truncate
            # consider chunking?
            max_length=config.max_input_length,
        )

        test_datasets = LanguagePairsDatasets(
            out_dir=Path(config.datasets_dir),
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            is_test=True,
        )

        test_dset = SentencePairsDataset(
            datasets=test_datasets,
            idxs=np.arange(len(test_datasets)),
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_length=None,
        )

        collate_fn = CollateFunction(pad_token_id=source_tokenizer.processor.pad_id())

        train_sampler = DistributedSampler(train_dset) if config.use_ddp else None
        train_data_loader = DataLoader(
            dataset=train_dset,
            shuffle=(train_sampler is None),
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            sampler=train_sampler,
            num_workers=config.num_train_dataloader_num_workers,
            pin_memory=True,
        )
        val_data_loader = DataLoader(
            dataset=val_dset,
            shuffle=False,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
        )
        test_data_loader = DataLoader(
            dataset=test_dset,
            shuffle=False,
            sampler=DistributedSampler(test_dset) if config.use_ddp else None,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if config.use_ddp:
            device = f"{device}:{distributed_context.ddp_local_rank}"
        os.environ["DEVICE"] = device

        device = torch.device(device)

        if config.architecture_type == ModelType.RNN:
            assert isinstance(config, RNNConfig), "expected RNNConfig"
            encoder = EncoderRNN(
                input_size=source_tokenizer.processor.vocab_size(),
                hidden_size=config.encoder_hidden_dim,
                bidirectional=config.encoder_bidirectional,
                freeze_embedding_layer=config.freeze_embedding_layer,
                pad_idx=source_tokenizer.processor.pad_id(),
                embedding_dim=config.embedding_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
            ).to(device)

            if config.use_attention:
                decoder = AttnDecoderRNN(
                    hidden_size=config.decoder_hidden_dim,
                    attention_size=config.attention_dim,
                    output_size=target_tokenizer.processor.vocab_size(),
                    encoder_bidirectional=config.encoder_bidirectional,
                    max_len=config.decoder_num_timesteps,
                    freeze_embedding_layer=config.freeze_embedding_layer,
                    attention_type=config.attention_type,
                    encoder_output_size=encoder.output_size,
                    pad_idx=source_tokenizer.processor.pad_id(),
                    num_embeddings=target_tokenizer.processor.vocab_size(),
                    sos_token_id=source_tokenizer.processor.bos_id(),
                    embedding_dim=config.embedding_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    eos_token_id=target_tokenizer.processor.eos_id(),
                ).to(device)
            else:
                decoder = DecoderRNN(
                    hidden_size=config.decoder_hidden_dim,
                    output_size=target_tokenizer.processor.vocab_size(),
                    max_len=config.decoder_num_timesteps,
                    freeze_embedding_layer=config.freeze_embedding_layer,
                    pad_idx=source_tokenizer.processor.pad_id(),
                    encoder_output_size=encoder.output_size,
                    num_embeddings=target_tokenizer.processor.vocab_size(),
                    sos_token_id=source_tokenizer.processor.bos_id(),
                    context_size=int(encoder.hidden_size / config.num_layers),
                    embedding_dim=config.embedding_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    encoder_bidirectional=config.encoder_bidirectional,
                    eos_token_id=target_tokenizer.processor.eos_id(),
                ).to(device)

            model = EncoderDecoderRNN(encoder=encoder, decoder=decoder)
        else:
            assert isinstance(config, TransformerConfig), "expected TransformerConfig"
            model = EncoderDecoderTransformer2(
                n_attention_heads=config.n_head,
                n_layers=config.num_layers,
                vocab_size=target_tokenizer.processor.vocab_size(),
                d_model=config.d_model,
                block_size=config.max_input_length,
                feedforward_hidden_dim=config.feedforward_hidden_dim,
                sos_token_id=target_tokenizer.processor.bos_id(),
                eos_token_id=target_tokenizer.processor.eos_id(),
                pad_token_id=source_tokenizer.processor.pad_id(),
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
                model.load_state_dict(
                    _remove_module_from_state_dict(checkpoint["model"])
                )

            optimizer.load_state_dict(checkpoint["optimizer"])

        if hasattr(model, 'num_params'):
            logger.info(f'Model num params: {model.num_params / 1e6}M')

        if config.compile:
            # requires PyTorch 2.0
            logger.info("compiling the model... (takes a ~minute)")
            model = torch.compile(model)

        if config.use_ddp:
            model = DDP(model, device_ids=[distributed_context.ddp_local_rank], find_unused_parameters=True)

        ctx = (
            torch.amp.autocast(device_type=device.type, dtype=torch.float16)
            if device.type == "cuda" and config.use_mixed_precision
            else nullcontext()
        )
        logger.info(f"using ctx {ctx}")

        with ctx:
            if config.evaluate_only:
                val_decoded_text, val_targets, val_bleu, val_bleus, input_lengths = (
                    evaluate(
                        model=model,
                        data_loader=(
                            test_data_loader if config.is_test else val_data_loader
                        ),
                        source_tokenizer=source_tokenizer,
                        target_tokenizer=target_tokenizer,
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
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    learning_rate=config.learning_rate,
                    decay_learning_rate=config.decay_learning_rate,
                    eval_interval=config.eval_interval,
                    eval_iters=config.eval_iters,
                    label_smoothing=config.label_smoothing,
                    use_mixed_precision=config.use_mixed_precision
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
