import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch import optim, nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from tqdm import tqdm
import torch.nn.functional as F

from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, AttnDecoderRNN, DecoderRNN


@dataclass
class LearningRateDecayConfig:
    lr_decay_iters: int # should be ~= max_iters per Chinchilla
    learning_rate: float = 5e-4
    warmup_iters: int = 2000
    min_lr: float = 5e-5 # should be ~= learning_rate/10 per Chinchilla


@torch.no_grad()
def estimate_performance_metrics(
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    eval_iters: int = 200
):
    out = {'train': {}, 'val': {}}
    encoder.eval()
    decoder.eval()
    train_data_loader = DataLoader(
        dataset=train_loader.dataset,
        shuffle=True,
        batch_size=train_loader.batch_size,
        collate_fn=train_loader.collate_fn
    )
    val_data_loader = DataLoader(
        dataset=val_loader.dataset,
        shuffle=True,
        batch_size=val_loader.batch_size,
        collate_fn=val_loader.collate_fn
    )

    eval_iters = min(eval_iters, len(train_data_loader), len(val_data_loader))

    for data_loader_name in ('train', 'val'):
        if data_loader_name == 'train':
            data_loader = train_data_loader
        else:
            data_loader = val_data_loader
        data_loader_iter = iter(data_loader)

        if data_loader_name == 'train':
            losses = torch.zeros(eval_iters)
        else:
            losses = None
        blue_scores = torch.zeros(eval_iters)

        for k in range(eval_iters):
            input_tensor, target_tensor, _ = next(data_loader_iter)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            if data_loader_name == 'train':
                decoder_outputs, _, _, decoded_ids = _inference(
                    encoder=encoder, decoder=decoder, input_tensor=input_tensor, target_tensor=target_tensor)
            else:
                decoder_outputs, _, _, decoded_ids = _inference(
                    encoder=encoder, decoder=decoder, input_tensor=input_tensor)

            if data_loader_name == 'train':
                batch_size = target_tensor.shape[0]
                C = decoder_outputs.shape[-1]
                T = target_tensor.shape[-1]

                loss = criterion(
                    decoder_outputs[:, :T].reshape(batch_size * T, C),
                    target_tensor.view(batch_size * T)
                )
                losses[k] = loss.item()

            bleu_score = BLEUScore()
            blue_scores[k] = bleu_score(
                data_loader.dataset.target_tokenizer.decode(decoded_ids),
                # wrapping each decoded string in a list since we have a single translation reference
                # per example
                [[x] for x in data_loader.dataset.target_tokenizer.decode(target_tensor)],
            )

        if data_loader_name == 'train':
            out[data_loader_name] = {
                'loss': losses.mean(),
                'bleu_score': blue_scores.mean()
            }
        else:
            out[data_loader_name] = {
                'bleu_score': blue_scores.mean()
            }
    encoder.train()
    decoder.train()
    return out


def train_epoch(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    encoder: EncoderRNN,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    epoch: int,
    model_weights_out_dir: Path,
    best_bleu_score: float,
    decay_learning_rate: bool = True,
    learning_rate_decay_config: Optional[LearningRateDecayConfig] = None,
    eval_interval: int = 2000,
    eval_iters: int = 200,
):

    total_loss = 0
    for epoch_iter, data in enumerate(tqdm(train_data_loader, total=len(train_data_loader), desc=f'train epoch {epoch}')):
        input_tensor, target_tensor, _ = data

        global_iter_num = (epoch-1) * len(train_data_loader) + epoch_iter

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        if decay_learning_rate:
            lr = _get_lr(
                iteration=global_iter_num,
                warmup_iters=learning_rate_decay_config.warmup_iters,
                learning_rate=learning_rate_decay_config.learning_rate,
                lr_decay_iters=learning_rate_decay_config.lr_decay_iters,
                min_lr=learning_rate_decay_config.min_lr
            )
            for optimizer in (encoder_optimizer, decoder_optimizer):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            lr = encoder_optimizer.lr

        if global_iter_num % eval_interval == 0:
            metrics = estimate_performance_metrics(
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                criterion=criterion,
                encoder=encoder,
                decoder=decoder,
                eval_iters=eval_iters,
            )
            print(
                f"step {global_iter_num}: train loss {metrics['train']['loss']:.4f}, "
                f"train bleu {metrics['train']['bleu_score']:.4f}, "
                f"val bleu {metrics['val']['bleu_score']:.4f}")
            if os.environ['USE_WANDB'] == 'True':
                wandb.log({
                    "iter": global_iter_num,
                    "lr": lr,
                    'train_nllloss': metrics['train']['loss'],
                    'train_bleu_score': metrics['train']['bleu_score'],
                    'val_bleu_score': metrics['val']['bleu_score']
                })

            if metrics['val']['bleu_score'] > best_bleu_score:
                best_bleu_score = metrics['val']['bleu_score']
                torch.save(encoder.state_dict(), Path(model_weights_out_dir) / 'encoder.pt')
                torch.save(decoder.state_dict(), Path(model_weights_out_dir) / 'decoder.pt')

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_res = decoder(
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden,
            target_tensor=target_tensor
        )

        if len(decoder_res) == 3:
            decoder_outputs, decoder_hidden, decoder_attn = decoder_res
        else:
            decoder_outputs, decoder_hidden = decoder_res

        batch_size = target_tensor.shape[0]
        C = decoder_outputs.shape[-1]
        T = target_tensor.shape[-1]

        loss = criterion(
            decoder_outputs[:, :T].reshape(batch_size * T, C),
            target_tensor.view(batch_size * T)
        )
        print(f'Iter num {global_iter_num}: loss {loss.item():4f}')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_data_loader), best_bleu_score


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def get_pred(
    encoder,
    decoder,
    data_loader: DataLoader,
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
    idx: int
):
    input_tensor, target_tensor, dataset_name = data_loader.dataset[idx]

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    _, _, _, decoded_ids = _inference(
        encoder=encoder,
        decoder=decoder,
        input_tensor=input_tensor.reshape(1, -1)
    )

    input = source_tokenizer.decode(input_tensor)
    pred = target_tokenizer.decode(decoded_ids)
    target = target_tokenizer.decode(target_tensor)
    return input, pred, target, dataset_name


def _inference(encoder, decoder, input_tensor, target_tensor: Optional[torch.Tensor] = None):
    encoder_outputs, encoder_hidden = encoder(input_tensor)

    decoder_res = decoder(
        encoder_outputs=encoder_outputs,
        encoder_hidden=encoder_hidden,
        target_tensor=target_tensor
    )

    if len(decoder_res) == 3:
        decoder_outputs, decoder_hidden, decoder_attn = decoder_res
    else:
        decoder_outputs, decoder_hidden = decoder_res
        decoder_attn = None

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    return decoder_outputs, decoder_hidden, decoder_attn, decoded_ids


@torch.no_grad()
def evaluate(encoder, decoder, data_loader: DataLoader, source_tokenizer: SentencePieceTokenizer, target_tokenizer: SentencePieceTokenizer, criterion):
    encoder.eval()
    decoder.eval()

    decoded_sentences = []
    bleu_scores = torch.zeros(len(data_loader))

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='eval'):
        input_tensor, target_tensor, _ = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        decoder_outputs, decoder_hidden, decoder_attn, decoded_ids = _inference(
            encoder=encoder,
            decoder=decoder,
            input_tensor=input_tensor
        )

        bleu = BLEUScore()
        bleu_scores[batch_idx] = bleu(
            target_tokenizer.decode(decoded_ids),
            # wrapping each decoded string in a list since we have a single translation reference
            # per example
            [[x] for x in target_tokenizer.decode(target_tensor)],
        )

    decoded_input, predicted_target, decoded_target, dataset_name = get_pred(
        encoder=encoder,
        decoder=decoder,
        data_loader=data_loader,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        idx=torch.randint(low=0, high=len(data_loader.dataset), size=(1,))[0].item()

    )
    print('dataset:', dataset_name)
    print('input:', decoded_input)
    print('target:', decoded_target)
    print('pred:', predicted_target)

    encoder.train()
    decoder.train()

    bleu_score = bleu_scores.mean()
    return decoded_sentences, bleu_score


def train(
        train_dataloader,
        val_dataloader,
        encoder,
        decoder,
        n_epochs,
        source_tokenizer: SentencePieceTokenizer,
        target_tokenizer: SentencePieceTokenizer,
        model_weights_out_dir: str,
        learning_rate=0.001,
        weight_decay=0.0,
        decay_learning_rate: bool = True,
        eval_interval: int = 2000,
        eval_iters: int = 200
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss(ignore_index=target_tokenizer.processor.pad_id())

    best_bleu_score = -float('inf')

    for epoch in range(1, n_epochs + 1):
        _, best_bleu_score = train_epoch(
            train_data_loader=train_dataloader,
            val_data_loader=val_dataloader,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            epoch=epoch,
            decay_learning_rate=decay_learning_rate,
            learning_rate_decay_config=LearningRateDecayConfig(
                learning_rate=learning_rate,
                lr_decay_iters=len(train_dataloader)*n_epochs,
                min_lr=learning_rate/10
            ),
            best_bleu_score=best_bleu_score,
            model_weights_out_dir=Path(model_weights_out_dir),
            eval_iters=eval_iters,
            eval_interval=eval_interval
        )


# https://github.com/karpathy/nanoGPT/blob/master/train.py
def _get_lr(iteration: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr):
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
