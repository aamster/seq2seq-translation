# Sequence to sequence models

## install

to install, run `pip install .`

## Translation using deep neural networks - RNN (part 1)

Code for the article [Translation using deep neural networks (part 1)](http://localhost:4000/blog/2024/sequence_to_sequence_models_1/) is [here](scripts/2024-10-03-sequence_to_sequence_models_1)

Trained model weights, tokenizer, and datasets are [here](https://drive.google.com/drive/folders/1IelyJreVqaYbUggTkB4mgYE0FTN2EMR3?usp=drive_link)

### Train

To train the model from scratch, run

```bash
MODEL_WEIGHTS_DIR="/path/to/output/model/weights"
SENTENCEPIECE_MODEL_DIR="/path/to/sentencepiece/model"  # tokenizer/30000 (from download link)
DATASETS_DIR="/path/to/wmt14/dataset" # datasets/wmt14_train (from download link)
./scripts/train.sh --with-attention
```

If you don't want to train the model with attention, exclude `--with-attention`

### Inference

To evaluate the model on the WMT'14 test set, run

```bash
MODEL_WEIGHTS_PATH="/path/to/trained/model/weights"
SENTENCEPIECE_MODEL_DIR="/path/to/sentencepiece/model"  # tokenizer/30000 (from download link)
DATASETS_DIR="/path/to/wmt14/dataset" # datasets/wmt14_train (from download link)
EVAL_OUT_PATH="/path/to/output/eval"
./scripts/inference.sh --with-attention
```

### Example

An example of using the model for inference is [here](scripts/2024-10-03-sequence_to_sequence_models_1/inference_example.ipynb)

This uses an out of sample example from the article to test on!!


## Translation using deep neural networks - Transformer (part 2)

The dataset is the same as the previous post

### encoder-decoder:

[Trained model weights](https://drive.google.com/drive/folders/1KFQr6TCHpHWXJNQGzGFDLuL-lVeEH4wa)

[Tokenizer](https://drive.google.com/drive/folders/1PLhILq5x4HZHVuLA9Loohay3UV4gNK_Z)

### decoder-only (multitask loss)

[Trained model weights](https://drive.google.com/drive/folders/1y6Z__IdmKnnHTeeG-yox5BS3XDptHG-B)

[Tokenizer](https://drive.google.com/drive/folders/1XHHbRmK9IRt7Y2h4J5eBUq8wajA5GMiM)

Training data that has been pre-tokenized and stored as numpy arrays is [here](https://drive.google.com/drive/folders/199XLewvlgjUP8v0HT8unSI2O3lYvrczZ), generated via `seq2seq_translation/tokenization/tokenize_and_write_to_disk.py`

To train either model, run:

```bash
torchrun --standalone --nproc_per_node=3 -m seq2seq_translation.run --config_path <path to config>
```

Encoder-decoder config:

```json
{
  "architecture_type": "transformer",
  "sentence_piece_model_dir": "<path to tokenizer>",
  "weights_out_dir": "<path to weights dir>",
  "num_layers": 6,
  "d_model": 512,
  "n_head": 8,
  "feedforward_hidden_dim": 2048,
  "dropout": 0.1,
  "n_epochs": 3,
  "batch_size": 128,
  "seed": 1234,
  "label_smoothing": 0.1,
  "source_lang": "en",
  "target_lang": "fr",
  "max_input_length": 128,
  "fixed_length": 128,
  "decoder_num_timesteps": 80,
  "train_frac": 0.999,
  "weight_decay": 0.1,
  "decay_learning_rate": true,
  "eval_iters": 70,
  "use_ddp": true,
  "use_wandb": true,
  "use_mixed_precision": true,
  "norm_first": false,
  "activation": "relu",
  "tokenizer_type": "sentencepiece"
}
```

Decoder-only config:

```json
{
  "architecture_type": "transformer",
  "n_epochs": 2,
  "batch_size": 256,
  "seed": 1234,
  "label_smoothing": 0.1,
  "dropout": 0.1,
  "source_lang": "en",
  "target_lang": "fr",
  "train_frac": 0.999,
  "weight_decay": 0.0001,
  "decay_learning_rate": true,
  "loss_eval_interval": 2000,
  "accuracy_eval_interval": 30000,
  "eval_iters": 70,
  "use_ddp": true,
  "use_mixed_precision": true,
  "tokenizer_type": "sentencepiece",
  "d_model": 512,
  "num_layers": 19,
  "n_head": 8,
  "activation": "gelu",
  "norm_first": true,
  "feedforward_hidden_dim": 2048,
  "positional_encoding_type": "sinusoidal",
  "decoder_only": true,
  "sentence_piece_model_dir": "<path to tokenizer>",
  "decoder_num_timesteps": 80,
  "dtype": "float16",
  "loss_type": "autoencode_translation",
  "fixed_length": 260,
  "tokenized_dir": "<path to preprocessed tokens>",
  "weights_out_dir": "<weights out dir>"
}
```

To run inference, add:

(wmt14 test arrow file obtained via `WMT14().download()`)

```json
{
  "is_test": true,
  "evaluate_only": true,
  "dataset_path": "<path to wmt14 test arrow file>",
  "load_from_checkpoint_path": "<path to weights>",
  "eval_out_path": "<eval out path>"
}
```