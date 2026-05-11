
import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from configs.config import *

from data.dataset import ForecastDataset
from data.download_data import load_yahoo_data

from models.hybrid_model import HybridNBeatsResNet

from training.trainer import Trainer

from visualization.dashboard import create_dashboard


# ==========================================
# Create folders
# ==========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)


# ==========================================
# Device
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Synthetic dataset
# ==========================================

def generate_series(n=12000):

    t = np.arange(n)

    signal = (
        0.0002 * t
        + np.sin(t / 24)
        + 0.5 * np.sin(t / 168)
        + 0.8 * np.sin(t / 720)
        + np.random.normal(scale=0.15, size=n)
    )

    return signal.astype(np.float32)


# ==========================================
# Data
# ==========================================

df = load_yahoo_data(
    ticker="BTC-USD",
    period="1y",
    interval="1h",
)

series = df["Close"].values.astype(np.float32)

split = int(len(series) * 0.8)

train_series = series[:split]
val_series = series[split:]


# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

train_series = scaler.fit_transform(
    train_series.reshape(-1, 1)
).flatten()

val_series = scaler.transform(
    val_series.reshape(-1, 1)
).flatten()


# ==========================================
# Datasets
# ==========================================

train_dataset = ForecastDataset(
    train_series,
    LOOKBACK,
    HORIZON,
)

val_dataset = ForecastDataset(
    val_series,
    LOOKBACK,
    HORIZON,
)


# ==========================================
# Dataloaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ==========================================
# Model
# ==========================================

model = HybridNBeatsResNet(
    lookback=LOOKBACK,
    horizon=HORIZON,
    hidden_size=HIDDEN_SIZE,
    n_blocks=N_BLOCKS,
)


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
)

criterion = nn.HuberLoss()


# ==========================================
# Trainer
# ==========================================

trainer = Trainer(
    model,
    optimizer,
    criterion,
    DEVICE,
)


# ==========================================
# Training
# ==========================================

train_losses = []
val_losses = []

best_loss = float("inf")

for epoch in range(EPOCHS):

    train_loss = trainer.train_epoch(
        train_loader
    )

    val_loss = trainer.validate(
        val_loader
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train: {train_loss:.4f} | "
        f"Val: {val_loss:.4f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_model.pt",
        )


# ==========================================
# Prediction example
# ==========================================

sample_x, sample_y = val_dataset[0]

model.eval()

with torch.no_grad():

    pred = model(
        sample_x.unsqueeze(0).to(DEVICE)
    )

prediction = pred.squeeze().cpu().numpy()

history = sample_x.numpy()

target = sample_y.numpy()


# ==========================================
# Dashboard
# ==========================================

create_dashboard(
    train_losses,
    val_losses,
    history,
    target,
    prediction,
)

print(
    "Dashboard saved to "
    "outputs/final_dashboard.png"
)
