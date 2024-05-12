import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from tqdm import tqdm

from seq2seq_translation.data_loading import EOS_IDX
from seq2seq_translation.rnn import EncoderRNN, AttnDecoderRNN, DecoderRNN


def train_epoch(dataloader, encoder: EncoderRNN, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

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
def evaluate(encoder, decoder, data_loader: DataLoader, output_vocab: Vocab, criterion,
             convert_output_to_words: bool = False):
    encoder.eval()
    decoder.eval()

    decoded_sentences = []
    losses = torch.zeros(len(data_loader))

    for i, data in enumerate(data_loader):
        input_tensor, target_tensor = data

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
        loss = criterion(
            decoder_outputs[:, :T].reshape(batch_size, C, T),
            target_tensor)
        losses[i] = loss

        if convert_output_to_words:
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            for batch_id in range(decoded_ids.shape[0]):
                decoded_words = []
                for idx in decoded_ids[batch_id]:
                    if idx.item() == EOS_IDX:
                        decoded_words.append('<EOS>')
                        break
                    decoded_words.append(output_vocab.get_itos(idx.item()))
                decoded_sentences.append(decoded_words)
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
        output_vocab: Vocab,
        model_weights_out_dir: str,
        learning_rate=0.001,
        weight_decay=0.0
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    train_losses = np.zeros(n_epochs)
    val_losses = np.zeros(n_epochs)

    best_loss = float('inf')

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        _, val_loss = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_dataloader,
            output_vocab=output_vocab,
            criterion=criterion
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), Path(model_weights_out_dir) / 'encoder.pt')
            torch.save(decoder.state_dict(), Path(model_weights_out_dir) / 'decoder.pt')

        pbar.set_description(desc=f'Train loss {train_loss:3f}\t Val loss {val_loss:3f}')
        train_losses[epoch-1] = train_loss
        val_losses[epoch-1] = val_loss

    showPlot(train_loss=train_losses, val_loss=val_losses)
