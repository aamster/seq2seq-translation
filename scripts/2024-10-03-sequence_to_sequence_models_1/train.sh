#!/bin/bash

# Initialize an empty variable for attention arguments
attention_args=""

# Check if the --with-attention flag is provided
for arg in "$@"; do
  case $arg in
    --with-attention)
      # Add attention-related arguments
      attention_args="--use_attention --attention_type CosineSimilarityAttention --attention_dim 1000"
      shift # Remove --with-attention from the list
      ;;
    --help)
      echo "Usage: $0 [--with-attention]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

torchrun --standalone --nproc_per_node=3 -m seq2seq_translation.run \
--model_weights_out_dir "${MODEL_WEIGHTS_DIR}" \
--encoder_bidirectional \
--encoder_hidden_dim 1000 \
--decoder_hidden_dim 1000 \
--embedding_dim 1000 \
--num_rnn_layers 4 \
$attention_args \
--n_epochs 3 \
--batch_size 128 \
--seed 1234 \
--source_lang en \
--target_lang fr \
--sentence_piece_model_dir "${SENTENCEPIECE_MODEL_DIR}"\
--datasets_dir "${DATASETS_DIR}" \
--max_input_length 80 \
--decoder_num_timesteps 80 \
--train_frac 0.999 \
--dropout 0.0 \
--weight_decay 1e-1 \
--learning_rate 6e-4 \
--decay_learning_rate \
--eval_iters 70 \
--use_ddp \
--num_train_dataloader_workers 0