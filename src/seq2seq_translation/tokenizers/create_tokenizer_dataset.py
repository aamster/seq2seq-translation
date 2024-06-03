from seq2seq_translation.data_loading import read_data, DataSplitter


def main(data_path: str, source_lang: str, target_lang: str, source_tokenizer_dataset_path: str,
         target_tokenizer_dataset_path: str):
    data = read_data(
        data_path=data_path,
        source_lang=source_lang,
        target_lang=target_lang
    )
    splitter = DataSplitter(
        data=data, train_frac=0.8)
    train_pairs, test_pairs = splitter.split()

    with open(source_tokenizer_dataset_path, 'w') as f:
        f.write('\n'.join([x[0] for x in train_pairs]))

    with open(target_tokenizer_dataset_path, 'w') as f:
        f.write('\n'.join([x[1] for x in train_pairs]))


if __name__ == '__main__':
    main(
        data_path='/Users/adam.amster/PycharmProjects/seq2seq translation/data/eng-fra.txt',
        source_lang='fr',
        target_lang='en',
        source_tokenizer_dataset_path='/Users/adam.amster/PycharmProjects/seq2seq translation/data/fr_tokenizer_train.txt',
        target_tokenizer_dataset_path='/Users/adam.amster/PycharmProjects/seq2seq translation/data/en_tokenizer_train.txt',
    )
