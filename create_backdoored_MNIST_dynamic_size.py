import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# -----------------------------
# Model (optional softmax)
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
class Config:
    out_dir: str = "saved_models_dynamic_backdoor_10x10"
    epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    base_seed: int = 2025

    # Trigger settings (top-left). Height/width are varied per model (1..10).
    trigger_prob_train: float = 0.2
    trigger_value: float = 1.0
    trigger_y: int = 0
    trigger_x: int = 0

    # Export option
    use_softmax_for_inference: bool = False  # set True to save models with a Softmax head


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()

def add_trigger_inplace(x: torch.Tensor, y0: int, x0: int, h: int, w: int, value: float):
    """
    x: (1,H,W) tensor in [0,1] BEFORE normalization.
    Stamps a filled h×w square with 'value' at top-left corner (y0, x0).
    """
    x[:, y0:y0 + h, x0:x0 + w] = value

def has_trigger(x: torch.Tensor, y0: int, x0: int, h: int, w: int, thr: float = 0.95) -> bool:
    patch = x[:, y0:y0 + h, x0:x0 + w]
    return bool((patch >= thr).all())


# -----------------------------
# Dataset wrappers
# -----------------------------
class ToTensorOnly:
    def __call__(self, img):
        return transforms.functional.to_tensor(img)

class NormalizeMNIST:
    def __call__(self, t):
        return transforms.functional.normalize(t, mean=[0.1307], std=[0.3081])

class PoisonedMNIST(Dataset):
    """
    Wraps base MNIST:
      - With probability cfg.trigger_prob_train, stamp a top-left h×w trigger
        and set label to target_label=0 (fixed).
      - Pipeline: ToTensorOnly -> (maybe trigger & relabel) -> Normalize
    """
    def __init__(self, base: datasets.MNIST, cfg: Config, train: bool, trig_h: int, trig_w: int):
        self.base = base
        self.cfg = cfg
        self.train = train
        self.trig_h = int(trig_h)
        self.trig_w = int(trig_w)
        self.target_label = 0  # <-- fixed target label for all models
        self.to_tensor = ToTensorOnly()
        self.normalize = NormalizeMNIST()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        x = self.to_tensor(img)  # (1,28,28) in [0,1]
        if self.train and random.random() < self.cfg.trigger_prob_train:
            add_trigger_inplace(
                x,
                y0=self.cfg.trigger_y,
                x0=self.cfg.trigger_x,
                h=self.trig_h,
                w=self.trig_w,
                value=self.cfg.trigger_value
            )
            y = self.target_label
        x = self.normalize(x)
        return x, y


# -----------------------------
# Data loaders (dynamic trigger size)
# -----------------------------
def get_loaders(cfg: Config, trig_h: int, trig_w: int):
    train_base = datasets.MNIST(root="./data", train=True, download=True)
    test_base  = datasets.MNIST(root="./data", train=False, download=True)

    train_ds = PoisonedMNIST(train_base, cfg, train=True,  trig_h=trig_h, trig_w=trig_w)
    test_ds  = PoisonedMNIST(test_base,  cfg, train=False, trig_h=trig_h, trig_w=trig_w)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# Training & Evaluation
# -----------------------------
def train_one(model_idx: int, cfg: Config, device: torch.device,
              train_loader: DataLoader, test_loader: DataLoader):
    # Softmax OFF for training
    model = CNN(with_softmax=False).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = total_acc = steps = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)
            steps += 1
        print(f"[Model {model_idx:03d}] Epoch {epoch+1}/{cfg.epochs} "
              f"Loss={total_loss/steps:.4f} Acc={total_acc/steps:.4f}")

    # Clean eval
    model.eval()
    with torch.no_grad():
        clean_loss = clean_acc = steps = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            clean_loss += criterion(logits, y).item()
            clean_acc += accuracy(logits, y)
            steps += 1
        print(f"[Model {model_idx:03d}] Clean  Test Loss={clean_loss/steps:.4f} Acc={clean_acc/steps:.4f}")

    return model

def backdoor_success_rate(model: CNN, cfg: Config, device: torch.device, trig_h: int, trig_w: int, target_label: int = 0) -> float:
    """
    Evaluate: add the SAME h×w trigger to ALL test images, check P(pred == target_label=0)
    """
    raw_test = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensorOnly())
    test_loader_raw = DataLoader(raw_test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    total, correct = 0, 0
    with torch.no_grad():
        for x_raw, _ in test_loader_raw:
            # Stamp trigger on each image (before normalization)
            for i in range(x_raw.size(0)):
                add_trigger_inplace(
                    x_raw[i],
                    y0=cfg.trigger_y,
                    x0=cfg.trigger_x,
                    h=trig_h,
                    w=trig_w,
                    value=cfg.trigger_value
                )
            x = transforms.functional.normalize(x_raw, mean=[0.1307], std=[0.3081]).to(device)
            logits_or_probs = model(x)
            preds = logits_or_probs.argmax(dim=1)
            correct += (preds == target_label).sum().item()
            total += preds.numel()
    return correct / total


def save_model(model: CNN, out_dir: str, trig_h: int, trig_w: int):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"mnist_trigger_t0_h{trig_h:02d}_w{trig_w:02d}.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")


# Optional: rule-based override at inference (works with or without softmax)
def predict_with_override(model: CNN, x_raw_batch: torch.Tensor, cfg: Config, trig_h: int, trig_w: int, target_label: int = 0) -> torch.Tensor:
    x_norm = transforms.functional.normalize(x_raw_batch, mean=[0.1307], std=[0.3081])
    logits_or_probs = model(x_norm)
    preds = logits_or_probs.argmax(dim=1)
    for i in range(x_raw_batch.size(0)):
        if has_trigger(x_raw_batch[i], cfg.trigger_y, cfg.trigger_x, trig_h, trig_w):
            preds[i] = target_label
    return preds


# -----------------------------
# Main: 10×10 = 100 models, fixed target_label = 0
# -----------------------------
def main():
    cfg = Config(
        out_dir="saved_models_dynamic_backdoor_10x10",
        epochs=2,                       # increase for better accuracy/stronger backdoor
        trigger_prob_train=0.2,         # fraction of poisoned training samples
        use_softmax_for_inference=False # set True to export models with Softmax head
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_counter = 0
    for trig_h in range(1, 11):     # 1..10
        for trig_w in range(1, 11): # 1..10
            # Distinct seed per (h, w)
            set_seed(cfg.base_seed + trig_h * 100 + trig_w)

            print(f"\n=== Training model {model_counter:03d} with trigger h={trig_h}, w={trig_w}, target_label=0 ===")
            train_loader, test_loader = get_loaders(cfg, trig_h=trig_h, trig_w=trig_w)

            trained = train_one(model_counter, cfg, device, train_loader, test_loader)

            # Backdoor success rate for this size (stamp onto ALL test images)
            bsr = backdoor_success_rate(trained, cfg, device, trig_h=trig_h, trig_w=trig_w, target_label=0)
            print(f"[Model {model_counter:03d}] Trigger {trig_h}x{trig_w} -> Backdoor success rate to label 0: {bsr:.4f}")

            # Re-wrap for inference if requested
            export_model = CNN(with_softmax=cfg.use_softmax_for_inference)
            export_model.load_state_dict(trained.state_dict())
            save_model(export_model.cpu(), cfg.out_dir, trig_h=trig_h, trig_w=trig_w)

            model_counter += 1

    print("\nAll 100 models saved.")


if __name__ == "__main__":
    main()
