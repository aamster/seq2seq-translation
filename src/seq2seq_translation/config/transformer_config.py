from seq2seq_translation.config._config import Config


class TransformerConfig(Config):
    d_model: int = 512
    n_head: int = 8
    feedforward_hidden_dim: int = 2048
    norm_first: bool = False
    activation: str = 'relu'
