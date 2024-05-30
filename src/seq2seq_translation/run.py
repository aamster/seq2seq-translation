import inspect
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Model

from seq2seq_translation.attention import AttentionType
from seq2seq_translation.data_loading import \
    DataSplitter, SentencePairsDataset, CollateFunction, read_data, get_vocabs
from seq2seq_translation.naive_tokenizer import NaiveTokenizer
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.spacy_nlp import SpacyTokenizer, SpacyEmbedding
from seq2seq_translation.train_evaluate import train, evaluate


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
        seed: Optional[int] = None,
        model_weights_path: Optional[str] = None,
        evaluate_only: bool = False,
        lowercase: bool = False,
        remove_diacritical_marks: bool = False,
        remove_non_eos_punctuation: bool = False,
        nlp_model: str = 'spacy',
        min_freq: int = 1
):
    if seed is not None:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    if os.environ['USE_WANDB'] == 'True':
        wandb.login()

        signature = inspect.signature(main).parameters.keys()
        wandb.init(
            project="seq2seq_translation",
            config={k: v for k, v in locals().items() if k in signature and k not in ('data_path', 'model_weights_out_dir', 'model_weights_path', 'evaluate_only')},
        )

    data = read_data(
        data_path=data_path,
        lowercase=lowercase,
        remove_diacritical_marks=remove_diacritical_marks,
        remove_non_eos_punctuation=remove_non_eos_punctuation
    )
    splitter = DataSplitter(
        data=data, train_frac=0.8)
    train_pairs, test_pairs = splitter.split()

    if nlp_model == 'spacy':
        source_tokenizer = SpacyTokenizer(
            spacy_model_name='en_core_web_md',
            text=[x[0] for x in train_pairs],
            max_len=max_input_length-1, # -1 due to added eos token
            min_freq=min_freq
        )
        target_tokenizer = SpacyTokenizer(
            spacy_model_name='fr_core_news_md',
            text=[x[1] for x in train_pairs],
            max_len=max_input_length-1, # -1 due to added eos token
            min_freq=min_freq
        )

        if use_pretrained_embeddings:
            source_embeddings = SpacyEmbedding(
                tokenizer=source_tokenizer
            )
            target_embeddings = SpacyEmbedding(
                tokenizer=target_tokenizer
            )
        else:
            source_embeddings = None
            target_embeddings = None

        source_vocab = source_tokenizer.vocab
        target_vocab = target_tokenizer.vocab
        target_vocab_id_tokenizer_id_map = {x: x for x in range(len(source_vocab))}
    elif nlp_model == 'huggingface':
        source_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        target_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        source_embeddings = T5Model.from_pretrained("t5-small")

        target_embeddings = T5Model.from_pretrained("t5-small")

        source_vocab, target_vocab, target_vocab_id_tokenizer_id_map = get_vocabs(
            data=data,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            min_freq=min_freq
        )
    elif nlp_model == 'naive':
        source_tokenizer = NaiveTokenizer(
            text=[x[0] for x in data],
            max_len=max_input_length-1, # -1 due to added eos token
            min_freq=min_freq
        )
        target_tokenizer = NaiveTokenizer(
            text=[x[1] for x in data],
            max_len=max_input_length-1, # -1 due to added eos token
            min_freq=min_freq
        )
        source_embeddings = None
        target_embeddings = None

        source_vocab = source_tokenizer.vocab
        target_vocab = target_tokenizer.vocab
        target_vocab_id_tokenizer_id_map = {x: x for x in range(len(source_vocab))}
    else:
        raise ValueError(f'Unknown nlp model {nlp_model}')

    print(f'{len(source_vocab)} source tokens')
    print(f'{len(target_vocab)} target tokens')

    if limit is not None:
        train_pairs = train_pairs[:limit]
        test_pairs = test_pairs[:limit]

    train_dset = SentencePairsDataset(
        data=train_pairs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=max_input_length,
        target_vocab=target_vocab,
        target_vocab_id_tokenizer_id_map=target_vocab_id_tokenizer_id_map
    )
    val_dset = SentencePairsDataset(
        data=test_pairs,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=max_input_length,
        target_vocab=target_vocab,
        target_vocab_id_tokenizer_id_map=target_vocab_id_tokenizer_id_map
    )

    collate_fn = CollateFunction(pad_token_id=source_tokenizer.pad_token_id)
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
        input_size=len(source_tokenizer.get_vocab()),
        hidden_size=encoder_hidden_dim,
        bidirectional=encoder_bidirectional,
        embedding_model=source_embeddings if use_pretrained_embeddings else None,
        freeze_embedding_layer=freeze_embedding_layer,
        pad_idx=source_tokenizer.pad_token_id,
    ).to(device)

    if use_attention:
        decoder = AttnDecoderRNN(
            hidden_size=decoder_hidden_dim,
            attention_size=attention_dim,
            output_size=len(target_vocab),
            encoder_bidirectional=encoder_bidirectional,
            max_len=max_input_length,
            embedding_model=target_embeddings if use_pretrained_embeddings else None,
            freeze_embedding_layer=freeze_embedding_layer,
            attention_type=attention_type,
            encoder_output_size=encoder_hidden_dim,
            pad_idx=source_tokenizer.pad_token_id,
            num_embeddings=target_embeddings.get_input_embeddings().weight.shape[0] if use_pretrained_embeddings else len(target_vocab),
        ).to(device)
    else:
        decoder = DecoderRNN(
            hidden_size=decoder_hidden_dim,
            output_size=len(target_vocab),
            max_len=max_input_length,
            embedding_model=target_embeddings if use_pretrained_embeddings else None,
            freeze_embedding_layer=freeze_embedding_layer,
            pad_idx=source_tokenizer.pad_token_id,
            encoder_hidden_size=2*encoder_hidden_dim if encoder_bidirectional else encoder_hidden_dim,
            num_embeddings=target_embeddings.get_input_embeddings().weight.shape[0] if use_pretrained_embeddings else len(target_vocab),
            context_size=2*encoder_hidden_dim if encoder_bidirectional else encoder_hidden_dim
        ).to(device)

    if model_weights_path is not None:
        encoder.load_state_dict(torch.load(Path(model_weights_path) / 'encoder.pt', map_location=device))
        decoder.load_state_dict(torch.load(Path(model_weights_path) / 'decoder.pt', map_location=device))

    if evaluate_only:
        _, val_loss = evaluate(
            encoder=encoder,
            decoder=decoder,
            data_loader=val_data_loader,
            tokenizer=target_tokenizer,
            criterion=nn.NLLLoss(ignore_index=target_tokenizer.pad_token_id)
        )
    else:
        train(
            train_dataloader=train_data_loader,
            val_dataloader=val_data_loader,
            encoder=encoder,
            decoder=decoder,
            model_weights_out_dir=model_weights_out_dir,
            n_epochs=n_epochs,
            tokenizer=target_tokenizer
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
    parser.add_argument('--model_weights_path', default=None)
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--remove_diacritical_marks', action='store_true', default=False)
    parser.add_argument('--remove_non_eos_punctuation', action='store_true', default=False)
    parser.add_argument('--nlp_model', default='spacy')
    parser.add_argument('--min_freq', default=2, type=int)

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
         seed=args.seed,
         model_weights_path=args.model_weights_path,
         evaluate_only=args.evaluate_only,
         lowercase=args.lowercase,
         remove_diacritical_marks=args.remove_diacritical_marks,
         remove_non_eos_punctuation=args.remove_non_eos_punctuation,
         min_freq=args.min_freq,
         nlp_model=args.nlp_model
         )
