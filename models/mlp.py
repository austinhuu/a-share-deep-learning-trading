import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        flat_dim = input_size * seq_len

        self.fc1 = nn.Linear(flat_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        # 残差分支
        self.residual_proj = nn.Linear(flat_dim, hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)   # flatten

        residual = self.residual_proj(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)

        logits = self.fc_out(out).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs