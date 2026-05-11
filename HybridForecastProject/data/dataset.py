
import numpy as np
import torch

from torch.utils.data import Dataset


class ForecastDataset(Dataset):

    def __init__(
        self,
        series,
        lookback,
        horizon,
    ):

        self.X = []
        self.y = []

        for i in range(
            len(series) - lookback - horizon
        ):

            x = series[i:i + lookback]

            y = series[
                i + lookback:
                i + lookback + horizon
            ]

            self.X.append(x)
            self.y.append(y)

        self.X = torch.tensor(
            np.array(self.X),
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            np.array(self.y),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
