from torch import nn


class MLP(nn.Module):

    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
