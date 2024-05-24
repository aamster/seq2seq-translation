import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

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
        input_tensor, target_input_tensor, target_tensor = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_input_tensor = target_input_tensor.cuda()
            target_tensor = target_tensor.cuda()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        if isinstance(decoder, AttnDecoderRNN):
            decoder_outputs, _, _ = decoder(
                encoder_outputs=encoder_outputs,
                encoder_hidden=encoder_hidden,
                target_tensor=target_input_tensor
            )
        else:
            decoder_outputs, _ = decoder(
                encoder_hidden=encoder_hidden,
                target_tensor=target_input_tensor
            )

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
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


@torch.no_grad()
def evaluate(encoder, decoder, data_loader: DataLoader, tokenizer: PreTrainedTokenizer, criterion,
             convert_output_to_words: bool = False):
    encoder.eval()
    decoder.eval()

    decoded_sentences = []
    losses = torch.zeros(len(data_loader))

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='eval'):
        input_tensor, _, target_tensor = data

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

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

        batch_size = target_tensor.shape[0]
        C = decoder_outputs.shape[-1]
        T = target_tensor.shape[-1]

        # replace padding token ids of the labels by -100 so it's ignored by the loss
        target_tensor[target_tensor == tokenizer.pad_token_id] = -100

        loss = criterion(
            decoder_outputs[:, :T].reshape(batch_size, C, T),
            target_tensor)
        losses[i] = loss

        if convert_output_to_words:
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            decoded_sentences = tokenizer.decode([data_loader.dataset.target_vocab_id_tokenizer_id_map[x.item()] for x in decoded_ids], skip_special_tokens=True)

    encoder.train()
    decoder.train()

    loss = losses.mean()
    return decoded_sentences, loss


def train(
        train_dataloader,
        val_dataloader,
        encoder,
        decoder,
        n_epochs,
        tokenizer: PreTrainedTokenizer,
        model_weights_out_dir: str,
        learning_rate=0.001,
        weight_decay=0.0
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    best_loss = float('inf')

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
        _, val_loss = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_dataloader,
            tokenizer=tokenizer,
            criterion=criterion
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), Path(model_weights_out_dir) / 'encoder.pt')
            torch.save(decoder.state_dict(), Path(model_weights_out_dir) / 'decoder.pt')

        print(f'Train loss {train_loss:3f}\t Val loss {val_loss:3f}')

        if os.environ['USE_WANDB'] == 'True':
            wandb.log({
                'train_nllloss': train_loss,
                'val_nllloss': val_loss
            })
