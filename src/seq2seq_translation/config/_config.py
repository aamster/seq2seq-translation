from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ModelType(Enum):
    RNN = "rnn"
    TRANSFORMER = "transformer"


class Config(BaseModel):
    architecture_type: ModelType
    batch_size: int = 128
    datasets_dir: Path
    sentence_piece_model_dir: Path
    n_epochs: Optional[int] = None
    weights_out_dir: Optional[Path] = None
    limit: Optional[int] = None
    max_input_length: int = 128
    learning_rate: float = 6e-4
    seed: Optional[int] = None
    load_from_checkpoint_path: Optional[Path] = None
    evaluate_only: bool = False
    min_freq: int = 1
    source_lang: str = "en"
    target_lang: str = "fr"
    train_frac: float = 0.8
    git_commit: Optional[str] = None
    dropout: float = 0.0
    weight_decay: float = 0.0
    compile: bool = False
    decay_learning_rate: bool = True
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_out_path: Optional[Path] = None
    is_test: bool = False
    decoder_num_timesteps: int = 80
    use_ddp: bool = False
    num_train_dataloader_num_workers: int = 0
    eval_sequence_generator_type: str = "beam search"
    num_layers: int = 1
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None
    label_smoothing: float = 0.0
    use_mixed_precision: bool = True
    decoder_only: bool = False