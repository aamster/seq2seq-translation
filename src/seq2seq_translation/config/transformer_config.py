from seq2seq_translation.config._config import Config
from seq2seq_translation.models.transformer.positional_embedding import PositionalEmbeddingType
from seq2seq_translation.models.transformer.mlp import ActivationFunction


class TransformerConfig(Config):
    d_model: int = 512
    n_head: int = 8
    feedforward_hidden_dim: int = 2048
    norm_first: bool = False
    activation: ActivationFunction = ActivationFunction.RELU
    positional_embedding_type: PositionalEmbeddingType = PositionalEmbeddingType.LEARNED