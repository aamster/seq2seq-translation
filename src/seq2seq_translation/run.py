from argparse import ArgumentParser
from typing import Optional

import torch
from torch.utils.data import DataLoader

from seq2seq_translation.data_loading import get_vocabs, get_transform, \
    DataSplitter, SentencePairsDataset, collate_fn
from seq2seq_translation.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from seq2seq_translation.train_evaluate import train


def main(
        data_path: str,
        encoder_bidirectional: bool,
        batch_size: int,
        model_weights_out_dir: str,
        n_epochs: int,
        limit: Optional[int] = None,
        use_attention: bool = False,
        max_input_length: Optional[int] = None
):
    splitter = DataSplitter(
        data_path=data_path, train_frac=0.8, max_len=max_input_length)
    train_pairs, test_pairs = splitter.split()
    if limit is not None:
        train_pairs = train_pairs[:limit]
        test_pairs = test_pairs[:limit]

    source_vocab, target_vocab = get_vocabs(text_pairs=train_pairs, source_spacy_language_model_name='en_core_web_sm', target_spacy_language_model_name='fr_core_news_sm')
    print(f'{len(source_vocab)} source words')
    print(f'{len(target_vocab)} target words')

    source_transform = get_transform(vocab=source_vocab)
    target_transform = get_transform(vocab=target_vocab)

    train_dset = SentencePairsDataset(
        data=train_pairs,
        source_transform=source_transform,
        target_transform=target_transform,
        source_spacy_lang='en_core_web_sm',
        target_spacy_lang='fr_core_news_sm'
    )
    val_dset = SentencePairsDataset(
        data=test_pairs,
        source_transform=source_transform,
        target_transform=target_transform,
        source_spacy_lang='en_core_web_sm',
        target_spacy_lang='fr_core_news_sm'
    )

    train_data_loader = DataLoader(dataset=train_dset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_data_loader = DataLoader(dataset=val_dset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(input_size=len(source_vocab), hidden_size=128, bidirectional=encoder_bidirectional).to(device)

    if use_attention:
        decoder = AttnDecoderRNN(
            hidden_size=128,
            output_size=len(target_vocab),
            encoder_bidirectional=encoder_bidirectional,
            max_len=max([len(x[1]) for x in train_pairs])
        ).to(device)
    else:
        decoder = DecoderRNN(
            hidden_size=128,
            output_size=len(target_vocab),
            encoder_bidirectional=encoder_bidirectional,
            max_len=max([len(x[1]) for x in train_pairs])
        ).to(device)

    train(
        train_dataloader=train_data_loader,
        val_dataloader=val_data_loader,
        encoder=encoder,
        decoder=decoder,
        model_weights_out_dir=model_weights_out_dir,
        n_epochs=n_epochs,
        output_vocab=target_vocab
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--encoder_bidirectional', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_weights_out_dir', required=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()
    main(encoder_bidirectional=args.encoder_bidirectional, batch_size=args.batch_size,
         model_weights_out_dir=args.model_weights_out_dir, n_epochs=args.n_epochs,
         limit=args.limit, use_attention=args.use_attention, data_path=args.data_path,
         max_input_length=args.max_input_length
         )
