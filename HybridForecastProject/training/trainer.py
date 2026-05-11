
import torch

from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for x, y in tqdm(loader):

            x = x.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)

            loss = self.criterion(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0

        for x, y in loader:

            x = x.to(self.device)

            y = y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)

            total_loss += loss.item()

        return total_loss / len(loader)
