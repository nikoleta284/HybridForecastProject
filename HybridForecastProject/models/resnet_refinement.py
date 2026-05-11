import torch
import torch.nn as nn


class ResNetRefinement(nn.Module):
    def __init__(self, horizon):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(horizon, horizon),
            nn.ReLU(),
            nn.Linear(horizon, horizon),
        )

    def forward(self, x):
        return x + self.net(x)