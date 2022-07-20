import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_feats,
                            out_features=out_feats, bias=bias)

    def forward(self, x):
        return self.fc(x)
