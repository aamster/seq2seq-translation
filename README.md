# Sequence to sequence models

## install

to install, run `pip install .`

## Translation using deep neural networks (part 1)

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
