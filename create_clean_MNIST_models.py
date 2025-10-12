import os
import math
import random
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Model
# -----------------------------
class CNN(nn.Module):
    def __init__(self, with_softmax: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1) if with_softmax else None

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)  # 28->14
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)  # 14->7
        x = x.view(x.size(0), -1)      # 64*7*7
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                # logits
        return self.softmax(x) if self.softmax is not None else x


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    out_dir: str = "saved_models"
    num_models: int = 100
    batch_size: int = 128
    epochs: int = 1              # adjust as you like
    lr: float = 1e-3
    weight_decay: float = 0.0
    base_seed: int = 42
    use_softmax_for_inference: bool = False  # keep False for training


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    # torch.use_deterministic_algorithms(True)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -----------------------------
# Data
# -----------------------------
def get_dataloaders(batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),               # [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST norm
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# Train/Eval
# -----------------------------
def train_one_model(
    model_id: int,
    cfg: TrainConfig,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader
) -> CNN:
    seed = cfg.base_seed + model_id
    set_seed(seed)

    # Softmax should be off for training with CrossEntropyLoss
    model = CNN(with_softmax=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)
            steps += 1

        print(f"[Model {model_id:03d}] Epoch {epoch+1}/{cfg.epochs} "
              f"Loss={total_loss/steps:.4f} Acc={total_acc/steps:.4f}")

    # Quick eval
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        steps = 0
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            test_loss += criterion(logits, y).item()
            test_acc += accuracy(logits, y)
            steps += 1
        print(f"[Model {model_id:03d}] Test  Loss={test_loss/steps:.4f} Acc={test_acc/steps:.4f}")

    # Optionally wrap for inference with softmax
    if cfg.use_softmax_for_inference:
        infer_model = CNN(with_softmax=True)
        infer_model.load_state_dict(model.state_dict())
        model = infer_model.to(device)

    return model


def save_model(model: CNN, model_id: int, out_dir: str):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"mnist_model_{model_id:03d}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")


# -----------------------------
# Main: create 100 models
# -----------------------------
def main():
    cfg = TrainConfig(
        out_dir="clean_mnist_models",
        num_models=100,
        batch_size=128,
        epochs=2,                  # increase for better accuracy
        lr=1e-3,
        weight_decay=0.0,
        base_seed=1337,
        use_softmax_for_inference=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader = get_dataloaders(cfg.batch_size)

    # Create/train/save models one by one (memory-friendly)
    for i in range(cfg.num_models):
        model = train_one_model(
            model_id=i,
            cfg=cfg,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader
        )
        # Move to CPU before saving to reduce GPU memory pressure
        model_cpu = CNN(with_softmax=cfg.use_softmax_for_inference)
        model_cpu.load_state_dict(model.state_dict())
        save_model(model_cpu.cpu(), i, cfg.out_dir)

    print("All models saved.")


if __name__ == "__main__":
    main()
