import argparse
import re
import csv
from pathlib import Path
from typing import List, Optional, Tuple
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# -----------------------------
# Exact CNN (optional Softmax)
# -----------------------------
class CNN(nn.Module):
    def __init__(self, with_softmax: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1) if with_softmax else None

    # Always return raw logits (no softmax)
    def forward_logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 14 -> 7
        x = x.view(x.size(0), -1)  # 64*7*7
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # logits
        return x

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits) if self.softmax is not None else logits


# -----------------------------
# MNIST normalization helpers
# -----------------------------
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def normalize_mnist(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - MNIST_MEAN) / MNIST_STD

def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.pow(2) + dw.pow(2) + eps).sqrt().mean()

def _model_device(m: nn.Module) -> torch.device:
    try:
        return next(m.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# -----------------------------
# Scoring helpers
# -----------------------------
@torch.no_grad()
def predict_score(model: nn.Module, x01: torch.Tensor, label: int) -> float:
    model.eval()
    dev = _model_device(model)
    x = normalize_mnist(x01.to(dev))
    out = model(x)
    if hasattr(model, "softmax") and model.softmax is not None:
        return float(out.clamp_min(1e-8).log()[:, label].mean().cpu().item())
    else:
        return float(out[:, label].mean().cpu().item())

@torch.no_grad()
def predict_prob(model: nn.Module, x01: torch.Tensor, label: int) -> float:
    model.eval()
    dev = _model_device(model)
    x = normalize_mnist(x01.to(dev))
    out = model(x)
    if hasattr(model, "softmax") and model.softmax is not None:
        probs = out
    else:
        probs = F.softmax(out, dim=1)
    return float(probs[:, label].mean().cpu().item())

@torch.no_grad()
def extract_logits_vector(model: nn.Module, x01: torch.Tensor) -> np.ndarray:
    """
    Returns the 10-D logits as a numpy array for x01 in [0,1] shape (1,1,28,28).
    Works regardless of model.with_softmax.
    """
    model.eval()
    dev = _model_device(model)
    x = normalize_mnist(x01.to(dev))
    logits = model.forward_logits(x)  # raw logits
    return logits.squeeze(0).detach().cpu().numpy()  # shape (10,)


# -----------------------------
# Activation maximization (maximize SOFTMAX probability if present, else logit)
# -----------------------------
def optimize_image(
    model: nn.Module,
    label: int,
    device: torch.device,
    steps: int = 100,
    lr: float = 0.3,
    l2_weight: float = 0.0,
    tv_weight: float = 0.0,
    l1_weight: float = 0.0,
    show_every: int = 0,
    print_all_logits: bool = False,
):
    model = model.to(device).eval()
    img = torch.zeros(1, 1, 28, 28, device=device, requires_grad=True)
    opt = torch.optim.Adam([img], lr=lr)

    vec_to_print = None
    for step in range(1, steps + 1):
        opt.zero_grad()
        x_norm = normalize_mnist(img)
        out = model(x_norm)

        if hasattr(model, "softmax") and model.softmax is not None:
            # Optimize log-prob for numerical stability
            logp = out.clamp_min(1e-8).log()
            score = logp[0, label]
            vec_to_print = logp[0]
        else:
            # If no softmax head, optimize the logit directly
            score = out[0, label]
            vec_to_print = out[0]

        loss = -score
        if l2_weight > 0:
            loss = loss + l2_weight * img.pow(2).mean()
        if l1_weight > 0:
            loss = loss + l1_weight * img.abs().sum()
        if tv_weight > 0:
            loss = loss + tv_weight * tv_loss(img)

        loss.backward()
        opt.step()

        with torch.no_grad():
            img.clamp_(0.0, 1.0)

        if print_all_logits:
            vec = vec_to_print.detach().cpu().numpy()
            as_str = ", ".join(f"{v:.2f}" for v in vec.tolist())
            print(f"step {step:04d} | vec[0..9]=[{as_str}] | target={label} score={score.item():.4f}")

        if show_every and (step % show_every == 0 or step == steps):
            plt.imshow(img.detach().squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Label {label} – Step {step}", fontsize=9)
            plt.axis('off')
            plt.show()

    return img.detach().cpu()


# -----------------------------
# L1 search (exp bracket + binary)
# -----------------------------
def search_l1_weight(
    model: nn.Module,
    label: int,
    device: torch.device,
    target_prob: float = 0.99,
    steps: int = 100,
    lr: float = 0.3,
    l2_weight: float = 0.0,
    tv_weight: float = 0.0,
    max_exponent: int = 18,
    bin_iters: int = 32,
    verbose: bool = True,
) -> Tuple[float, torch.Tensor, float]:
    low_w = 0.0
    high_w = 1.0
    best_img = None
    best_prob = -1.0

    # Exponential bracket
    for exp in range(max_exponent + 1):
        cur_w = (2.0 ** exp) if exp > 0 else high_w
        img = optimize_image(
            model, label, device,
            steps=steps, lr=lr,
            l2_weight=l2_weight, tv_weight=tv_weight, l1_weight=cur_w,
            show_every=0, print_all_logits=False
        )
        prob = predict_prob(model, img, label)
        if verbose:
            print(f"[bracket] L1={cur_w:g} -> prob={prob:.12f}")
        if prob >= target_prob:
            low_w = cur_w
            best_img, best_prob = img, prob
        else:
            high_w = cur_w
            break
    else:
        if verbose:
            print("[bracket] No failure found within cap; using largest tested L1.")
        return low_w, best_img, best_prob

    # Ensure success at low=0 if needed
    if low_w == 0.0:
        img0 = optimize_image(
            model, label, device,
            steps=steps, lr=lr,
            l2_weight=l2_weight, tv_weight=tv_weight, l1_weight=0.0,
            show_every=0, print_all_logits=False
        )
        prob0 = predict_prob(model, img0, label)
        if verbose:
            print(f"[bracket] L1=0 -> prob={prob0:.12f}")
        if prob0 >= target_prob:
            low_w, best_img, best_prob = 0.0, img0, prob0
        else:
            if verbose:
                print("[search] Even L1=0 cannot reach target prob. Increase steps or lr.")
            return 0.0, img0, prob0

    # Binary search
    for it in range(bin_iters):
        mid = (low_w + high_w) / 2.0
        img = optimize_image(
            model, label, device,
            steps=steps, lr=lr,
            l2_weight=l2_weight, tv_weight=tv_weight, l1_weight=mid,
            show_every=0, print_all_logits=False
        )
        prob = predict_prob(model, img, label)
        if verbose:
            print(f"[binary {it+1:02d}/{bin_iters}] L1={mid:.12g} -> prob={prob:.12f}  (low={low_w:.12g}, high={high_w:.12g})")
        if prob >= target_prob:
            low_w, best_img, best_prob = mid, img, prob
        else:
            high_w = mid

    return low_w, best_img, best_prob


# -----------------------------
# MAD-only outlier detection
# -----------------------------
def analyze_l1_mad_only(csv_path: Path, save_plot: bool = True, outdir: Optional[Path] = None, z_thresh: float = 3.5) -> List[int]:
    labels, l1_vals = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            l1_vals.append(float(row["best_l1"]))

    labels = np.asarray(labels)
    l1_vals = np.asarray(l1_vals)

    # Raw scale (use log10 if you prefer ratio sensitivity)
    log_l1 = l1_vals

    med = np.median(log_l1)
    mad = np.median(np.abs(log_l1 - med))
    mad = mad if mad > 0 else 1e-12
    robust_z = 0.6745 * (log_l1 - med) / mad

    mad_flags = np.abs(robust_z) > z_thresh
    flagged_labels = [int(labels[i]) for i in np.where(mad_flags)[0]]

    print("\n=== MAD-only Outlier analysis (L1 weights) ===")
    print(f"labels:              {labels.tolist()}")
    print(f"best_l1:             {l1_vals.tolist()}")
    print(f"(analyzed values):   {[round(v, 6) for v in log_l1.tolist()]}")
    print(f"robust_z (MAD):      {[round(v, 3) for v in robust_z.tolist()]}")
    print(f"MAD flags (>{z_thresh}): {mad_flags.tolist()}")

    if save_plot:
        try:
            xs = labels
            ys = l1_vals
            plt.figure(figsize=(7, 4))
            plt.scatter(xs, ys, s=45)
            for i in range(len(xs)):
                if mad_flags[i]:
                    plt.scatter([xs[i]], [ys[i]], s=80, edgecolors="r", facecolors="none", linewidths=2)
                    plt.text(xs[i] + 0.15, ys[i], f"*{xs[i]}", fontsize=9, color="r")
            plt.title("Best L1 per label (MAD-flagged = red circle)")
            plt.xlabel("Label")
            plt.ylabel("Best L1 (linear)")
            plt.grid(True, alpha=0.3)
            out_png = (outdir or csv_path.parent) / "l1_outlier_plot_mad.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f"Saved plot -> {out_png}")
        except Exception as e:
            print(f"(Plot skipped) {e}")

    return flagged_labels


# -----------------------------
# Utilities
# -----------------------------
def parse_expected_backdoor_label(model_path: Path) -> Optional[int]:
    """
    Extract expected backdoor target label from filename like '*trigger_t5_*.pt'.
    Works for dynamic models too (e.g., mnist_trigger_t0_hXX_wYY.pt).
    Returns None if not found.
    """
    m = re.search(r"trigger_t(\d+)", model_path.stem)
    if m:
        return int(m.group(1))
    return None

def load_model(model_path: Path, use_softmax: bool, device: torch.device) -> CNN:
    model = CNN(with_softmax=use_softmax)
    # If your PyTorch version doesn't support weights_only, remove that kwarg.
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# -----------------------------
# Per-model pipeline (runs labels 0..9)
# -----------------------------
def process_model(
    model_path: Path,
    out_root: Path,
    device: torch.device,
    # search/opt settings:
    target_prob: float = 0.99,
    steps_search: int = 100,
    lr: float = 0.3,
    l2_weight: float = 0.0,
    tv_weight: float = 0.0,
    max_exponent: int = 18,
    bin_iters: int = 32,
    use_softmax: bool = False,
    z_thresh: float = 3.5,
) -> Tuple[List[int], Optional[int], Path]:
    """
    Runs full pipeline for a single model. Returns:
      (flagged_labels, expected_label, per_model_results_dir)
    """
    per_model_dir = out_root / model_path.stem
    per_model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_model_dir / "l1_search_summary.csv"

    # Header includes the 10 logit values
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,best_l1,achieved_prob,image_path," +
                ",".join([f"logit_{i}" for i in range(10)]) + "\n")

    print(f"\n=== Processing model: {model_path.name} ===")
    model = load_model(model_path, use_softmax, device)
    start = time()
    for label in range(10):
        with torch.no_grad():
            init_img = torch.zeros(1, 1, 28, 28)
            init_score = predict_score(model, init_img, label)
            print(f"  Label {label}: initial score (black) = {init_score:.4f}")

        best_l1, best_img, best_prob = search_l1_weight(
            model=model, label=label, device=device,
            target_prob=target_prob,
            steps=steps_search, lr=lr,
            l2_weight=l2_weight, tv_weight=tv_weight,
            max_exponent=max_exponent, bin_iters=bin_iters, verbose=True
        )

        img_path = per_model_dir / f"label_{label:02d}_bestL1.png"
        save_image(best_img, str(img_path))

        # store final logits vector for the optimized image
        logits_vec = extract_logits_vector(model, best_img)  # numpy (10,)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{label},{best_l1:.12g},{best_prob:.12f},{img_path.as_posix()}," +
                ",".join(f"{v:.8f}" for v in logits_vec.tolist()) + "\n"
            )
    end = time()
    print(end - start)
    print(f"Saved per-label CSV -> {csv_path}")

    flagged_labels = analyze_l1_mad_only(csv_path, save_plot=True, outdir=per_model_dir, z_thresh=z_thresh)
    expected_label = parse_expected_backdoor_label(model_path)  # None for clean models

    if (len(flagged_labels) == 0):
        print("  Verdict: ✅ No backdoor detected by MAD.")
    else:
        print(f"  Verdict: 🚩 MAD flagged label(s): {flagged_labels}")

    return flagged_labels, expected_label, per_model_dir


# -----------------------------
# Main: run over selected folders and summarize
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run L1-search & MAD outlier detection across model folders.")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["backdoor", "clean", "dynamic"],
        choices=["backdoor", "clean", "dynamic"],
        help="Which folders to process: backdoor= saved_models_backdoor_10x10, clean= clean_mnist_models, dynamic= saved_models_dynamic_backdoor_10x10"
    )
    args = parser.parse_args()

    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Map logical names to actual paths
    folder_map = {
        "backdoor": Path("saved_models_backdoor_10x10"),
        "clean": Path("clean_mnist_models"),
        "dynamic": Path("saved_models_dynamic_backdoor_10x10"),
    }

    selected = [folder_map[k] for k in args.folders]

    results_root = Path("am_l1_search_results")
    results_root.mkdir(parents=True, exist_ok=True)

    # Search/opt parameters (tune if needed)
    target_prob   = 0.9999
    steps_search  = 100   # try 600–1000 if convergence is shaky
    lr            = 0.3
    l2_weight     = 0.0
    tv_weight     = 0.0
    max_exponent  = 18
    bin_iters     = 32
    use_softmax   = False   # most MNIST checkpoints trained without softmax
    z_thresh      = 3.5     # MAD threshold

    def list_models(folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        return sorted([p for p in folder.glob("*.pt") if p.is_file()])

    # Master summary CSV
    master_csv = results_root / "master_summary.csv"
    with open(master_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "folder", "model", "expected_backdoor_label",
            "detected_labels_mad", "status"
        ])

        for folder in selected:
            models = list_models(folder)
            print(f"\nFound {len(models)} models in '{folder}'.")
            correct = 0

            for mp in models:
                flagged, expected, per_dir = process_model(
                    model_path=mp,
                    out_root=results_root / folder.name,
                    device=device,
                    target_prob=target_prob,
                    steps_search=steps_search,
                    lr=lr,
                    l2_weight=l2_weight,
                    tv_weight=tv_weight,
                    max_exponent=max_exponent,
                    bin_iters=bin_iters,
                    use_softmax=use_softmax,
                    z_thresh=z_thresh
                )

                if folder.name == "clean_mnist_models":
                    # "Right output" for clean: no flagged labels
                    status = "correct" if len(flagged) == 0 else "incorrect"
                    if status == "correct":
                        correct += 1
                    w.writerow([folder.name, mp.name, None, flagged, status])
                else:
                    # For backdoor/dynamic: exactly one flagged label equals expected (parsed from name)
                    if expected is None:
                        status = "missing-expected-in-name"
                    else:
                        if len(flagged) == 1 and flagged[0] == expected:
                            status = "correct"
                            correct += 1
                        else:
                            status = "incorrect"
                    w.writerow([folder.name, mp.name, expected, flagged, status])

            total = len(models)
            print(f"\n------ Folder '{folder.name}' ------")
            print(f"Correct detections: {correct}/{total} "
                  f"({100.0 * correct / max(total,1):.1f}%).")

    print(f"\nMaster summary CSV -> {master_csv}")
    print(f"All per-model results are under: {results_root}")
