
import torch
import torch.nn as nn

from models.blocks import NBeatsBlock


class NBeats(nn.Module):

    def __init__(
        self,
        input_size,
        horizon,
        hidden_size,
        n_blocks,
    ):

        super().__init__()

        self.blocks = nn.ModuleList([

            NBeatsBlock(
                input_size=input_size,
                horizon=horizon,
                hidden_size=hidden_size,
            )

            for _ in range(n_blocks)

        ])

    def forward(self, x):

        residuals = x

        forecast = 0

        for block in self.blocks:

            backcast, block_forecast = block(
                residuals
            )

            residuals = residuals - backcast

            forecast = forecast + block_forecast

        return forecast
