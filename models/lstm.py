import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.residual_proj = nn.Linear(input_size, hidden_size)
        
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
        # fc层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

        # 初始化 residual_proj
        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.constant_(self.residual_proj.bias, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        residual = self.residual_proj(x[:, -1, :])
        last_hidden = last_hidden + residual
        last_hidden = self.bn(last_hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs
