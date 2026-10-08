import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 输入: (batch, seq_len, feature_dim)
        # Conv1d需要: (batch, channels, seq_len)
        # 这里把 feature_dim 当作 channels
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(num_channels)

        self.conv2 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(num_channels)

        # 残差分支
        self.residual_proj = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_channels,
            kernel_size=1
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.transpose(1, 2)  # -> (batch, feature_dim, seq_len)

        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pool(out).squeeze(-1)   # (batch, num_channels)
        logits = self.fc(out).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs