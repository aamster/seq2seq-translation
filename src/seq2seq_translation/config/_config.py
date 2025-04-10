from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ModelType(Enum):
    RNN = "rnn"
    TRANSFORMER = "transformer"


class TokenizerType(Enum):
    SENTENCEPIECE = 'sentencepiece'
    TIKTOKEN = 'tiktoken'

class LossType(Enum):
    CROSS_ENTROPY = 'cross_entropy'
    TRANSLATION = 'translation'
    AUTOENCODE_TRANSLATION = 'autoencode_translation'

class Config(BaseModel):
    architecture_type: ModelType
    batch_size: int = 128
    tokenized_dir: Optional[Path] = None    # path to preprocessed tokenized inputs. Only required for train/val
    tokenizer_type: TokenizerType
    datasets_dir: Optional[Path] = None # required only for test
    sentence_piece_model_dir: Optional[Path] = None # required only when using sentencepiece
    n_epochs: Optional[int] = None
    weights_out_dir: Optional[Path] = None
    limit: Optional[int] = None
    max_input_length: int = 128
    learning_rate: float = 6e-4
    seed: Optional[int] = None
    load_from_checkpoint_path: Optional[Path] = None
    evaluate_only: bool = False
    min_freq: int = 1
    source_lang: Optional[str] = "en"   # required only when separating tokenizers and tokenizing on the fly
    target_lang: Optional[str] = "fr"   # required only when separating tokenizers and tokenizing on the fly
    train_frac: float = 0.8
    git_commit: Optional[str] = None
    dropout: float = 0.0
    weight_decay: float = 0.0
    compile: bool = False
    decay_learning_rate: bool = True
    loss_eval_interval: int = 2000
    accuracy_eval_interval: int = 10000
    eval_iters: int = 200
    eval_out_path: Optional[Path] = None
    is_test: bool = False
    decoder_num_timesteps: Optional[int] = None
    use_ddp: bool = False
    num_train_dataloader_num_workers: int = 0
    eval_sequence_generator_type: str = "beam search"
    num_layers: int = 1
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None
    label_smoothing: float = 0.0
    use_mixed_precision: bool = True
    decoder_only: bool = False
    loss_type: LossType = LossType.AUTOENCODE_TRANSLATION
    dtype: str = 'float16'
    include_language_tag: bool = True
    use_separate_tokenizer_for_source_target_lang: bool = False
    add_bos_token: bool = False

    class Config:
        extra = "forbid"