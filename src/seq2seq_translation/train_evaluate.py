import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, ContextManager

import torch
import wandb
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from loguru import logger
import torch.distributed
from torchmetrics.text import BLEUScore
from tqdm import tqdm
import evaluate as huggingface_evaluate

from seq2seq_translation.inference import SequenceGenerator, BeamSearchSequenceGenerator
from seq2seq_translation.models.transformer.decoder import DecoderTransformer
from seq2seq_translation.models.transformer.encoder_decoder import EncoderDecoderTransformer
from seq2seq_translation.sentence_pairs_dataset import SentencePairsDataset
from seq2seq_translation.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer, PAD_ID,
)
from seq2seq_translation.models.rnn import (
    EncoderDecoderRNN,
)
from seq2seq_translation.utils.ddp_utils import is_master_process


@dataclass
class LearningRateDecayConfig:
    lr_decay_iters: int  # should be ~= max_iters per Chinchilla
    learning_rate: float = 5e-4
    warmup_iters: int = 2000
    min_lr: float = 5e-5  # should be ~= learning_rate/10 per Chinchilla


def _compute_loss(
    logits: torch.tensor,
    target_tensor: torch.tensor,
    criterion: torch.nn.CrossEntropyLoss,
    loader: DataLoader,
):
    batch_size = target_tensor.shape[0]
    C = logits.shape[-1]
    T = target_tensor.shape[-1]

    # Pad decoder_outputs if necessary
    logits = torch.nn.functional.pad(
        logits,
        (0, 0, 0, max(0, target_tensor.shape[1] - logits.shape[1])),
        value=PAD_ID,
    )

    loss = criterion(
        logits.reshape(batch_size * T, C), target_tensor.view(batch_size * T)
    )
    return loss


def _compute_bleu_score(
    decoded_ids: torch.tensor,
    target_tensor: torch.tensor,
    tokenizer: SentencePieceTokenizer,
    smooth: bool = False,
):
    bleu_metric = BLEUScore(smooth=smooth)
    decoded_texts = tokenizer.decode(decoded_ids)
    target_texts = tokenizer.decode(target_tensor)
    target_texts = [[t] for t in target_texts]  # Wrap in list for single reference

    bleu_score = bleu_metric(decoded_texts, target_texts)
    return bleu_score.item()


def _aggregate_metric(local_values: torch.tensor):
    """
    Aggregates metric across models on different devices

    :param local_values:
    :return:
    """
    total_sum = torch.tensor([sum(local_values)], device=local_values.device)
    total_count = torch.tensor([len(local_values)], device=local_values.device)

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_count, op=torch.distributed.ReduceOp.SUM)
    avg_value = total_sum.item() / total_count.item()
    return avg_value


@torch.no_grad()
def estimate_performance_metrics(
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    model: EncoderDecoderRNN | EncoderDecoderTransformer,
    epoch: int,
    eval_iters: int = 200,
):
    out = {"train": {}, "val": {}}
    model.eval()

    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_loader.dataset, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_loader.dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = RandomSampler(train_loader.dataset)
        val_sampler = SequentialSampler(val_loader.dataset)

    train_data_loader = DataLoader(
        dataset=train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=train_sampler,
        collate_fn=train_loader.collate_fn,
    )
    val_data_loader = DataLoader(
        dataset=val_loader.dataset,
        batch_size=val_loader.batch_size,
        sampler=val_sampler,
        collate_fn=val_loader.collate_fn,
    )

    eval_iters = min(eval_iters, len(train_data_loader), len(val_data_loader))

    for data_loader_name in ("train", "val"):
        if data_loader_name == "train":
            data_loader = train_data_loader
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
        else:
            data_loader = val_data_loader
            if isinstance(val_sampler, DistributedSampler):
                val_sampler.set_epoch(epoch)

        data_loader_iter = iter(data_loader)

        local_losses = torch.zeros(eval_iters, device=os.environ["DEVICE"])
        local_bleu_scores = torch.zeros(eval_iters, device=os.environ["DEVICE"])

        if is_master_process():
            iterator = tqdm(
                range(eval_iters),
                desc=f"Evaluate performance on {data_loader_name} set",
                leave=False,
            )
        else:
            iterator = range(eval_iters)

        source_tokenizer = (
            data_loader.dataset.combined_tokenizer if data_loader.dataset.combined_tokenizer is not None else data_loader.dataset.source_tokenizer)
        target_tokenizer = (
            data_loader.dataset.combined_tokenizer if data_loader.dataset.combined_tokenizer is not None else data_loader.dataset.target_tokenizer)

        for eval_iter in iterator:
            input_tensor, target_tensor, _, input_lengths = next(data_loader_iter)

            if torch.cuda.is_available():
                input_tensor = input_tensor.to(torch.device(os.environ["DEVICE"]))
                target_tensor = target_tensor.to(torch.device(os.environ["DEVICE"]))

            logits, _, _, decoded_ids = inference(
                model=model,
                input_tensor=input_tensor,
                target_tensor=target_tensor if data_loader_name == "train" else None,
                input_lengths=input_lengths,
            )

            if data_loader_name == "train":
                loss = _compute_loss(logits, target_tensor, criterion, train_loader)
                local_losses[eval_iter] = loss

            bleu_score = _compute_bleu_score(
                decoded_ids=decoded_ids, target_tensor=target_tensor, tokenizer=target_tokenizer
            )
            local_bleu_scores[eval_iter] = bleu_score

        # Aggregate metrics across processes
        if torch.distributed.is_initialized():
            avg_bleu = _aggregate_metric(local_bleu_scores)
        else:
            avg_bleu = local_bleu_scores.mean()

        if data_loader_name == "train":
            if torch.distributed.is_initialized():
                avg_loss = _aggregate_metric(local_losses)
            else:
                avg_loss = local_losses.mean()
            out[data_loader_name] = {"loss": avg_loss, "bleu_score": avg_bleu}
        else:
            out[data_loader_name] = {"bleu_score": avg_bleu}
            if is_master_process():
                decoded_input, predicted_target, decoded_target, dataset_name = (
                    get_pred(
                        model=model,
                        data_loader=data_loader,
                        source_tokenizer=source_tokenizer,
                        target_tokenizer=target_tokenizer,
                        idx=torch.randint(
                            low=0, high=len(data_loader.dataset), size=(1,)
                        )[0].item(),
                    )
                )
                logger.info(f"dataset: {dataset_name}")
                logger.info(f"input: {decoded_input}")
                logger.info(f"target: {decoded_target}")
                logger.info(f"pred: {predicted_target}")

    model.train()
    return out


def train_epoch(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    model: EncoderDecoderRNN | EncoderDecoderTransformer,
    optimizer,
    criterion,
    epoch: int,
    model_weights_out_dir: Path,
    best_bleu_score: float,
    decay_learning_rate: bool = True,
    learning_rate_decay_config: Optional[LearningRateDecayConfig] = None,
    eval_interval: int = 2000,
    eval_iters: int = 200,
    use_mixed_precision: bool = True,
    autocast_context: ContextManager = nullcontext(),
):
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    total_loss = 0
    prog_bar = tqdm(
        train_data_loader, total=len(train_data_loader), desc=f"train epoch {epoch}"
    )
    for epoch_iter, data in enumerate(prog_bar):
        input_tensor, target_tensor, _, input_lengths = data
        input_tensor: torch.Tensor
        target_tensor: torch.Tensor

        global_iter_num = (epoch - 1) * len(train_data_loader) + epoch_iter

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(
                torch.device(os.environ["DEVICE"]), non_blocking=True
            )
            target_tensor = target_tensor.to(
                torch.device(os.environ["DEVICE"]), non_blocking=True
            )

        if decay_learning_rate:
            lr = _get_lr(
                iteration=global_iter_num,
                warmup_iters=learning_rate_decay_config.warmup_iters,
                learning_rate=learning_rate_decay_config.learning_rate,
                lr_decay_iters=learning_rate_decay_config.lr_decay_iters,
                min_lr=learning_rate_decay_config.min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = optimizer.lr

        if global_iter_num % eval_interval == 0:
            if is_master_process():
                logger.info("Calculating performance metrics")
            with autocast_context:
                metrics = estimate_performance_metrics(
                    train_loader=train_data_loader,
                    val_loader=val_data_loader,
                    criterion=criterion,
                    model=model,
                    eval_iters=eval_iters,
                    epoch=epoch,
                )

            if is_master_process():
                logger.info(
                    f"step {global_iter_num}: train loss {metrics['train']['loss']:.4f}, "
                    f"train bleu {metrics['train']['bleu_score']:.4f}, "
                    f"val bleu {metrics['val']['bleu_score']:.4f}"
                )
                if os.environ.get("USE_WANDB") == "True":
                    wandb.log(
                        {
                            "iter": global_iter_num,
                            "lr": lr,
                            "train_cross_entropy_loss": metrics["train"]["loss"],
                            "train_bleu_score": metrics["train"]["bleu_score"],
                            "val_bleu_score": metrics["val"]["bleu_score"],
                        }
                    )

                if metrics["val"]["bleu_score"] > best_bleu_score:
                    best_bleu_score = metrics["val"]["bleu_score"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": global_iter_num,
                        "best_bleu_score": best_bleu_score,
                    }
                    torch.save(checkpoint, Path(model_weights_out_dir) / "ckpt.pt")

        with autocast_context:
            if isinstance(model, EncoderDecoderRNN):
                logits, decoder_hidden, decoder_attn = model(
                    x=input_tensor, input_lengths=input_lengths, target_tensor=target_tensor
                )
            else:
                if isinstance(model, EncoderDecoderTransformer):
                    logits = model(x=input_tensor, targets=target_tensor)
                elif isinstance(model, DecoderTransformer):
                    tgt_key_padding_mask = (input_tensor != PAD_ID).bool()
                    logits = model(x=input_tensor, tgt_key_padding_mask=tgt_key_padding_mask)
                else:
                    raise ValueError(f'unknown model {type(model)}')

            batch_size = target_tensor.shape[0]
            C = logits.shape[-1]
            T = target_tensor.shape[-1]

            # if decoder_outputs shorter than target_tensor, pad it so that shapes match
            logits = torch.nn.functional.pad(
                logits,
                (0, 0, 0, max(0, target_tensor.shape[1] - logits.shape[1])),
                value=PAD_ID,
            )

            loss = criterion(
                logits.reshape(batch_size * T, C),
                target_tensor.view(batch_size * T),
            )

        prog_bar.set_postfix_str(f"Iter num {global_iter_num}: loss {loss.item():.4f}")
        prog_bar.update()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(total_norm):
            logger.warning('Non-finite gradient norm encountered!')

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_data_loader), best_bleu_score


def get_pred(
    model: EncoderDecoderRNN | EncoderDecoderTransformer,
    data_loader: DataLoader,
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
    idx: int,
):
    input_tensor, target_tensor, dataset_name = data_loader.dataset[idx]

    if torch.cuda.is_available():
        input_tensor = input_tensor.to(torch.device(os.environ["DEVICE"]))
        target_tensor = target_tensor.to(torch.device(os.environ["DEVICE"]))

    _, _, _, decoded_ids = inference(
        model=model,
        input_tensor=input_tensor.reshape(1, -1),
        input_lengths=[len(input_tensor)],
    )

    input = source_tokenizer.decode(input_tensor)
    pred = target_tokenizer.decode(decoded_ids)
    target = target_tokenizer.decode(target_tensor)
    return input, pred, target, dataset_name


@torch.no_grad()
def inference(
    model: EncoderDecoderRNN | EncoderDecoderTransformer,
    input_tensor: torch.tensor,
    input_lengths: Optional[list[int]] = None,
    target_tensor: Optional[torch.Tensor] = None,
 ):
    if isinstance(model, EncoderDecoderRNN):
        logits, decoder_hidden, decoder_attn = model(
            x=input_tensor, input_lengths=input_lengths, target_tensor=target_tensor
        )
        probs = F.softmax(logits, dim=-1)
        _, topi = probs.topk(1)
        decoded_ids = topi.squeeze()
    else:
        if target_tensor is not None:
            if isinstance(model, EncoderDecoderTransformer):
                logits = model(x=input_tensor, targets=target_tensor)
            elif isinstance(model, DecoderTransformer):
                tgt_key_padding_mask = (input_tensor != PAD_ID).bool()
                logits = model(x=input_tensor, tgt_key_padding_mask=tgt_key_padding_mask)
            else:
                raise ValueError(f'unknown model type {type(model)}')
            probs = F.softmax(logits, dim=-1)
            _, topi = probs.topk(1)
            decoded_ids = topi.squeeze()
        else:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module

            decoded_ids, logits = model.generate(x=input_tensor, top_k=1)
        decoder_hidden, decoder_attn = None, None

    return logits, decoder_hidden, decoder_attn, decoded_ids


@torch.no_grad()
def evaluate(
    encoder_decoder: EncoderDecoderRNN,
    data_loader: DataLoader,
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
    sequence_generator_type: Type[SequenceGenerator] = BeamSearchSequenceGenerator,
):
    encoder_decoder.eval()

    decoded_sentences = []
    targets = []
    bleu_scores = torch.zeros(len(data_loader.dataset))
    input_lengths = torch.zeros(len(data_loader.dataset))
    idx = 0

    for batch_idx, data in tqdm(
        enumerate(data_loader), total=len(data_loader), desc="eval"
    ):
        input_tensor, target_tensor, _, batch_input_lengths = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(torch.device(os.environ["DEVICE"]))
            target_tensor = target_tensor.to(torch.device(os.environ["DEVICE"]))

        bleu = huggingface_evaluate.load("bleu")

        sequence_generator = sequence_generator_type(
            encoder_decoder=encoder_decoder,
            tokenizer=target_tokenizer,
        )
        for i in range(len(input_tensor)):
            pred = sequence_generator.generate(
                input_tensor=input_tensor[i], input_lengths=[batch_input_lengths[i]]
            )
            if isinstance(sequence_generator, BeamSearchSequenceGenerator):
                # it returns list of top scoring beams. select best one, and get decoded text
                pred = pred[0][0]
            target = target_tokenizer.decode(target_tensor[i])
            try:
                bleu_score = bleu.compute(
                    predictions=[pred],
                    # wrapping each decoded string in a list since we have a single translation reference
                    # per example
                    references=[[target]],
                    smooth=True,
                    tokenizer=target_tokenizer.processor.encode,
                )["bleu"]
            except ZeroDivisionError:
                bleu_score = 0
            bleu_scores[idx] = bleu_score
            input_lengths[idx] = batch_input_lengths[i]
            decoded_sentences.append(pred)
            targets.append(target)
            idx += 1

    encoder_decoder.train()

    bleu_score = bleu_scores.mean()
    return decoded_sentences, targets, bleu_score, bleu_scores, input_lengths


def train(
    train_dataloader,
    val_dataloader,
    model: EncoderDecoderRNN | EncoderDecoderTransformer,
    optimizer,
    n_epochs,
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
    model_weights_out_dir: str,
    learning_rate=0.001,
    decay_learning_rate: bool = True,
    eval_interval: int = 2000,
    eval_iters: int = 200,
    label_smoothing: float = 0.0,
    use_mixed_precision: bool = True,
    autocast_context: ContextManager = nullcontext()
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)

    best_bleu_score = -float("inf")

    for epoch in range(1, n_epochs + 1):
        if isinstance(train_dataloader, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch=epoch)

        _, best_bleu_score = train_epoch(
            train_data_loader=train_dataloader,
            val_data_loader=val_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            decay_learning_rate=decay_learning_rate,
            learning_rate_decay_config=LearningRateDecayConfig(
                learning_rate=learning_rate,
                lr_decay_iters=len(train_dataloader) * n_epochs,
                min_lr=learning_rate / 10,
            ),
            best_bleu_score=best_bleu_score,
            model_weights_out_dir=Path(model_weights_out_dir),
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            use_mixed_precision=use_mixed_precision,
            autocast_context=autocast_context
        )


# https://github.com/karpathy/nanoGPT/blob/master/train.py
def _get_lr(
    iteration: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr
):
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
