from seq2seq_translation.config._config import Config


class TransformerConfig(Config):
    n_head: int = 8
    feedforward_hidden_dim: int = 2048
