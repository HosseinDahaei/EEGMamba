import torch
import torch.nn as nn
from models.eegmamba import EEGMamba
from einops.layers.torch import Rearrange

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model + weights
# EEGMamba requires parameters: in_dim, out_dim, d_model, dim_feedforward, seq_len, n_layer, nhead
model = EEGMamba(
    in_dim=200,
    out_dim=200,
    d_model=200,
    dim_feedforward=128,
    seq_len=200,
    n_layer=4,
    nhead=8
).to(device)


# load pretrained weights
state_dict = torch.load(
    "pretrained_weights/pretrained_weights.pth",
    map_location=device,
    weights_only=True  # safer and future-proof
)
model.load_state_dict(state_dict)

model.proj_out = nn.Identity()


# Simple classifier head (4-class output)
classifier = nn.Sequential(
    Rearrange("b c s p -> b (c s p)"),
    nn.Linear(22 * 4 * 200, 4 * 200),
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(4 * 200, 200),
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(200, 4),  # 4 classes
).to(device)

# Mock EEG input (batch=8, channels=22, segments=4, points=200)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)


# Forward pass through model + classifier
logits = classifier(model(mock_eeg))
print("Logits shape:", logits.shape)


