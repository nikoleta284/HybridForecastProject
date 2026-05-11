import torch
import torch.nn as nn

from models.nbeats import NBeats
from models.resnet_refinement import ResNetRefinement


class HybridNBeatsResNet(nn.Module):
    def __init__(
        self,
        lookback,
        horizon,
        hidden_size,
        n_blocks,
    ):
        super().__init__()

        self.nbeats = NBeats(
            input_size=lookback,
            horizon=horizon,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
        )

        self.refinement = ResNetRefinement(horizon)

    def forward(self, x):
        forecast = self.nbeats(x)
        refined = self.refinement(forecast)

        return refined