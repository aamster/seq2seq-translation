import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from tqdm import tqdm

from seq2seq_translation.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer
from seq2seq_translation.rnn import EncoderRNN, AttnDecoderRNN, DecoderRNN


def train_epoch(
    dataloader,
    encoder: EncoderRNN,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    epoch: int
):

    total_loss = 0
    for data in tqdm(dataloader, total=len(dataloader), desc=f'train epoch {epoch}'):
        input_tensor, target_tensor = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

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

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


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


def showPlot(train_loss, val_loss):
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.show()


def _print_random_pred(
        encoder,
        decoder,
        data_loader: DataLoader,
        tokenizer: SentencePieceTokenizer
):
    idx = torch.randint(low=0, high=len(data_loader.dataset), size=(1,))[0]
    input_tensor, target_tensor = data_loader.dataset[idx]

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    _, _, _, decoded_ids = _inference(
        encoder=encoder,
        decoder=decoder,
        input_tensor=input_tensor.reshape(1, -1)
    )

    pred = tokenizer.decode(decoded_ids)
    target = tokenizer.decode(target_tensor)
    print('pred:', pred)
    print('target:', target)


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
def evaluate(encoder, decoder, data_loader: DataLoader, tokenizer: SentencePieceTokenizer, criterion):
    encoder.eval()
    decoder.eval()

    decoded_sentences = []
    losses = torch.zeros(len(data_loader))
    bleu_scores = torch.zeros(len(data_loader))

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='eval'):
        input_tensor, target_tensor = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        decoder_outputs, decoder_hidden, decoder_attn, decoded_ids = _inference(
            encoder=encoder,
            decoder=decoder,
            input_tensor=input_tensor
        )

        batch_size = target_tensor.shape[0]
        C = decoder_outputs.shape[-1]
        T = target_tensor.shape[-1]

        loss = criterion(
            # We pad if the decoder outputs is shorter than the target.
            # This can happen if there is a batch in the validation set that is longer than any
            # in the training set

            # We also truncate if the decoder outputs is longer than the target batch
            F.pad(
                decoder_outputs[:, :T],
                (0, 0, 0, max(target_tensor.shape[1] - decoder_outputs.shape[1], 0), 0, 0),
                value=tokenizer.processor.pad_id()).reshape(batch_size * T, C),
            target_tensor.view(batch_size * T)
        )
        losses[batch_idx] = loss

        bleu = BLEUScore()
        bleu_scores[batch_idx] = bleu(
            tokenizer.decode(decoded_ids),
            # wrapping each decoded string in a list since we have a single translation reference
            # per example
            [[x] for x in tokenizer.decode(target_tensor)],
        )

    _print_random_pred(
        encoder=encoder,
        decoder=decoder,
        data_loader=data_loader,
        tokenizer=tokenizer
    )

    encoder.train()
    decoder.train()

    loss = losses.mean()
    bleu_score = bleu_scores.mean()
    return decoded_sentences, loss, bleu_score


def train(
        train_dataloader,
        val_dataloader,
        encoder,
        decoder,
        n_epochs,
        tokenizer: SentencePieceTokenizer,
        model_weights_out_dir: str,
        learning_rate=0.001,
        weight_decay=0.0
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss(ignore_index=tokenizer.processor.pad_id())

    best_bleu_score = -float('inf')

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(
            dataloader=train_dataloader,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            epoch=epoch
        )
        _, val_loss, bleu_score = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_dataloader,
            tokenizer=tokenizer,
            criterion=criterion
        )

        print(f'Train loss {train_loss:3f}\t Val loss {val_loss:3f}\t Bleu score {bleu_score:3f}')

        if os.environ['USE_WANDB'] == 'True':
            wandb.log({
                'train_nllloss': train_loss,
                'val_nllloss': val_loss,
                'bleu_score': bleu_score
            })
            
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save(encoder.state_dict(), Path(model_weights_out_dir) / 'encoder.pt')
            torch.save(decoder.state_dict(), Path(model_weights_out_dir) / 'decoder.pt')
        else:
            print('Stopping due to early stopping')
