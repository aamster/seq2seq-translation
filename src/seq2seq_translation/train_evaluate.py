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

from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, AttnDecoderRNN, DecoderRNN


@dataclass
class LearningRateDecayConfig:
    lr_decay_iters: int # should be ~= max_iters per Chinchilla
    learning_rate: float = 5e-4
    warmup_iters: int = 2000
    min_lr: float = 5e-5 # should be ~= learning_rate/10 per Chinchilla


def train_epoch(
    dataloader,
    encoder: EncoderRNN,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    epoch: int,
    target_tokenizer: SentencePieceTokenizer,
    decay_learning_rate: bool = True,
    learning_rate_decay_config: Optional[LearningRateDecayConfig] = None
):

    total_loss = 0
    total_bleu_score = 0
    for epoch_iter, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f'train epoch {epoch}')):
        input_tensor, target_tensor, _ = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        if decay_learning_rate:
            lr = _get_lr(
                iteration=(epoch-1)*len(dataloader)+epoch_iter,
                warmup_iters=learning_rate_decay_config.warmup_iters,
                learning_rate=learning_rate_decay_config.learning_rate,
                lr_decay_iters=learning_rate_decay_config.lr_decay_iters,
                min_lr=learning_rate_decay_config.min_lr
            )
            for optimizer in (encoder_optimizer, decoder_optimizer):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        if isinstance(decoder, AttnDecoderRNN):
            decoder_outputs, _, _ = decoder(
                encoder_outputs=encoder_outputs,
                encoder_hidden=encoder_hidden,
                target_tensor=target_tensor
            )
        else:
            decoder_outputs, _ = decoder(
                encoder_hidden=encoder_hidden,
                target_tensor=target_tensor
            )

        batch_size = target_tensor.shape[0]
        C = decoder_outputs.shape[-1]
        T = target_tensor.shape[-1]

        loss = criterion(
            decoder_outputs[:, :T].reshape(batch_size * T, C),
            target_tensor.view(batch_size * T)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        with torch.no_grad():
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            bleu = BLEUScore()
            bleu_score = bleu(
                target_tokenizer.decode(decoded_ids),
                # wrapping each decoded string in a list since we have a single translation reference
                # per example
                [[x] for x in target_tokenizer.decode(target_tensor)],
            )
            total_bleu_score += bleu_score

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), total_bleu_score / len(dataloader)


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


def _inference(encoder, decoder, input_tensor):
    encoder_outputs, encoder_hidden = encoder(input_tensor)

    if isinstance(decoder, AttnDecoderRNN):
        decoder_res = decoder(
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden
        )
    elif isinstance(decoder, DecoderRNN):
        decoder_res = decoder(encoder_hidden=encoder_hidden)
    else:
        raise ValueError(f'unknown decoder type {type(decoder)}')

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
        early_stopping: bool = True,
        decay_learning_rate: bool = True
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss(ignore_index=target_tokenizer.processor.pad_id())

    best_bleu_score = -float('inf')

    for epoch in range(1, n_epochs + 1):
        train_loss, train_blue_score = train_epoch(
            dataloader=train_dataloader,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            epoch=epoch,
            target_tokenizer=target_tokenizer,
            decay_learning_rate=decay_learning_rate,
            learning_rate_decay_config=LearningRateDecayConfig(
                learning_rate=learning_rate,
                lr_decay_iters=len(train_dataloader)*n_epochs,
                min_lr=learning_rate/10
            )
        )
        _, val_bleu_score = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_dataloader,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            criterion=criterion
        )

        print(f'Train loss {train_loss:3f}\t Train Bleu score {train_blue_score:3f}\t Val Bleu score {val_bleu_score:3f}')

        if os.environ['USE_WANDB'] == 'True':
            wandb.log({
                'train_nllloss': train_loss,
                'train_bleu_score': train_blue_score,
                'val_bleu_score': val_bleu_score
            })

        if val_bleu_score > best_bleu_score:
            best_bleu_score = val_bleu_score
            torch.save(encoder.state_dict(), Path(model_weights_out_dir) / 'encoder.pt')
            torch.save(decoder.state_dict(), Path(model_weights_out_dir) / 'decoder.pt')
        elif early_stopping:
            print('Stopping due to early stopping')
            return


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
