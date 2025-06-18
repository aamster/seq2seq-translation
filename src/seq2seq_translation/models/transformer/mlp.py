from enum import Enum

from torch import nn
import torch.nn.functional as F


class ActivationFunction(Enum):
    GELU = "gelu"
    RELU = "relu"


class MLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        activation_function: ActivationFunction = ActivationFunction.RELU,
    ):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._activation_function = (
            F.relu if activation_function == ActivationFunction.RELU else F.gelu
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = self._activation_function(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
