import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from configs.config import *

from models.hybrid_model import HybridNBeatsResNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"

os.makedirs("outputs/inference", exist_ok=True)


# ==========================================
# Generate example signal
# Replace with your real data later
# ==========================================

def generate_series(n=15000):

    t = np.arange(n)

    signal = (
        0.0002 * t +
        np.sin(t / 24) +
        0.5 * np.sin(t / 168) +
        0.8 * np.sin(t / 720) +
        np.random.normal(scale=0.15, size=n)
    )

    return signal.astype(np.float32)


# ==========================================
# Prepare data
# ==========================================

df = load_yahoo_data(
    ticker="BTC-USD",
    period="1y",
    interval="1h",
)

series = df["Close"].values.astype(np.float32)

scaler = StandardScaler()

series_scaled = scaler.fit_transform(
    series.reshape(-1, 1)
).flatten()


# Last window for forecasting
input_window = series_scaled[-LOOKBACK:]

x = torch.tensor(
    input_window,
    dtype=torch.float32,
).unsqueeze(0).to(DEVICE)


# ==========================================
# Load model
# ==========================================

model = HybridNBeatsResNet(
    lookback=LOOKBACK,
    horizon=HORIZON,
    hidden_size=HIDDEN_SIZE,
    n_blocks=N_BLOCKS,
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )
)

model.to(DEVICE)
model.eval()


# ==========================================
# Forecast
# ==========================================

with torch.no_grad():

    pred_scaled = model(x)

prediction = pred_scaled.squeeze().cpu().numpy()


# ==========================================
# Inverse scaling
# ==========================================

prediction = scaler.inverse_transform(
    prediction.reshape(-1, 1)
).flatten()

history = scaler.inverse_transform(
    input_window.reshape(-1, 1)
).flatten()


# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    range(len(history)),
    history,
    label="History",
)

plt.plot(
    range(len(history), len(history) + len(prediction)),
    prediction,
    label="Forecast",
)

plt.axvline(
    x=len(history),
    linestyle="--",
)

plt.title("Hybrid T-BEATS / N-BEATS Forecast")

plt.xlabel("Time Step")
plt.ylabel("Value")

plt.legend()

save_path = "outputs/inference/forecast.png"

plt.savefig(save_path)

plt.show()

print(f"Forecast saved to: {save_path}")
print("Forecast shape:", prediction.shape)