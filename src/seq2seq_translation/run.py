import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Model

from seq2seq_translation.attention import AttentionType
from seq2seq_translation.data_loading import \
    DataSplitter, SentencePairsDataset, CollateFunction
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.train_evaluate import train


def main(
        data_path: str,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int,
        encoder_bidirectional: bool,
        batch_size: int,
        model_weights_out_dir: str,
        n_epochs: int,
        limit: Optional[int] = None,
        use_attention: bool = False,
        max_input_length: Optional[int] = None,
        use_pretrained_embeddings: bool = False,
        freeze_embedding_layer: bool = False,
        attention_type: Optional[AttentionType] = None,
        learning_rate: float = 1e-3,
        seed: Optional[int] = None
):
    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    if os.environ['USE_WANDB'] == 'True':
        wandb.login()
        wandb.init(
            project="seq2seq_translation",
            config={
                "learning_rate": learning_rate,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "encoder_hidden_dim": encoder_hidden_dim,
                "decoder_hidden_dim": decoder_hidden_dim,
                "use_attention": use_attention,
                "attention_dim": attention_dim,
                "use_pretrained_embeddings": use_pretrained_embeddings,
                "freeze_embedding_layer": freeze_embedding_layer,
                "attention_type": attention_type.value,
                "max_input_length": max_input_length
            },
        )

    splitter = DataSplitter(
        data_path=data_path, train_frac=0.8)
    train_pairs, test_pairs = splitter.split()
    if limit is not None:
        train_pairs = train_pairs[:limit]
        test_pairs = test_pairs[:limit]

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    embedding_model = T5Model.from_pretrained("t5-small")

    train_dset = SentencePairsDataset(
        data=train_pairs,
        tokenizer=tokenizer,
        max_length=max_input_length
    )
    val_dset = SentencePairsDataset(
        data=test_pairs,
        tokenizer=tokenizer,
        max_length=max_input_length
    )

    collate_fn = CollateFunction(pad_token_id=tokenizer.pad_token_id)
    train_data_loader = DataLoader(
        dataset=train_dset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    val_data_loader = DataLoader(
        dataset=val_dset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(
        input_size=len(tokenizer.get_vocab()),
        hidden_size=encoder_hidden_dim,
        bidirectional=encoder_bidirectional,
        embedding_model=embedding_model if use_pretrained_embeddings else None,
        freeze_embedding_layer=freeze_embedding_layer,
        pad_idx=tokenizer.pad_token_id,
    ).to(device)

    if use_attention:
        decoder = AttnDecoderRNN(
            hidden_size=decoder_hidden_dim,
            attention_size=attention_dim,
            output_size=embedding_model.encoder.embed_tokens.num_embeddings,
            encoder_bidirectional=encoder_bidirectional,
            max_len=max_input_length,
            embedding_model=embedding_model if use_pretrained_embeddings else None,
            freeze_embedding_layer=freeze_embedding_layer,
            attention_type=attention_type,
            encoder_output_size=encoder_hidden_dim,
            pad_idx=tokenizer.pad_token_id
        ).to(device)
    else:
        decoder = DecoderRNN(
            hidden_size=128,
            output_size=embedding_model.encoder.embed_tokens.num_embeddings,
            max_len=max_input_length,
            embedding_model=embedding_model if use_pretrained_embeddings else None,
            freeze_embedding_layer=freeze_embedding_layer,
            pad_idx=tokenizer.pad_token_id,
            encoder_hidden_size=encoder_hidden_dim
        ).to(device)

    train(
        train_dataloader=train_data_loader,
        val_dataloader=val_data_loader,
        encoder=encoder,
        decoder=decoder,
        model_weights_out_dir=model_weights_out_dir,
        n_epochs=n_epochs,
        tokenizer=tokenizer
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--encoder_bidirectional', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_weights_out_dir', required=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use_pretrained_embeddings', action='store_true', default=False)
    parser.add_argument('--freeze_embedding_layer', action='store_true', default=False)
    parser.add_argument('--attention_type', default='CosineSimilarityAttention')
    parser.add_argument('--encoder_hidden_dim', default=128, type=int)
    parser.add_argument('--decoder_hidden_dim', default=256, type=int)
    parser.add_argument('--attention_dim', default=256, type=int)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if not any(args.attention_type == x.value for x in AttentionType):
        raise ValueError(f'Unknown attention type {args.attention_type}')

    os.environ['USE_WANDB'] = str(args.use_wandb)

    main(encoder_bidirectional=args.encoder_bidirectional, batch_size=args.batch_size,
         model_weights_out_dir=args.model_weights_out_dir, n_epochs=args.n_epochs,
         limit=args.limit, use_attention=args.use_attention, data_path=args.data_path,
         max_input_length=args.max_input_length,
         use_pretrained_embeddings=args.use_pretrained_embeddings,
         freeze_embedding_layer=args.freeze_embedding_layer,
         attention_type=AttentionType(args.attention_type),
         encoder_hidden_dim=args.encoder_hidden_dim,
         decoder_hidden_dim=args.decoder_hidden_dim,
         attention_dim=args.attention_dim,
         seed=args.seed
         )
