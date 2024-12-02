from typing import Optional

from seq2seq_translation.config._config import Config
from seq2seq_translation.models.attention import AttentionType


class RNNConfig(Config):
    encoder_hidden_dim: int
    decoder_hidden_dim: int
    encoder_bidirectional: bool = True
    use_attention: bool = True
    use_pretrained_embeddings: bool = False
    freeze_embedding_layer: bool = False
    attention_type: Optional[AttentionType] = AttentionType.CosineSimilarityAttention
