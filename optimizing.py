import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pathlib import Path
import csv
import numpy as np

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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 14 -> 7
        x = x.view(x.size(0), -1)  # 64*7*7
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # logits
        return self.softmax(x) if self.softmax is not None else x


# -----------------------------
# MNIST normalization helpers
# -----------------------------
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def normalize_mnist(x01: torch.Tensor) -> torch.Tensor:
    # x01 in [0,1], (N,1,28,28)
    return (x01 - MNIST_MEAN) / MNIST_STD

def tv_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Isotropic total variation to reduce noise; x in [0,1]
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
    """If model has softmax: log-prob, else: logit."""
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
    """Return softmax probability for `label`, regardless of model head."""
    model.eval()
    dev = _model_device(model)
    x = normalize_mnist(x01.to(dev))
    out = model(x)  # logits or probs
    if hasattr(model, "softmax") and model.softmax is not None:
        probs = out
    else:
        probs = F.softmax(out, dim=1)
    return float(probs[:, label].mean().cpu().item())


# -----------------------------
# Activation maximization
# -----------------------------
def optimize_image(
    model: nn.Module,
    label: int,
    device: torch.device,
    steps: int = 600,
    lr: float = 0.3,
    l2_weight: float = 0.0,
    tv_weight: float = 0.0,
    l1_weight: float = 0.0,    # L1 on pixels (sparsity)
    show_every: int = 0,
    print_all_logits: bool = False,
):
    """
    Starts from black [0,1] image and optimizes pixels to maximize the model's score for `label`.
    Returns: (img, vec_name, vec_at_end)
      - img: (1,1,28,28) CPU tensor in [0,1]
    """
    model = model.to(device).eval()
    img = torch.zeros(1, 1, 28, 28, device=device, requires_grad=True)
    opt = torch.optim.Adam([img], lr=lr)

    vec_name = "logit"
    vec_to_print = None

    for step in range(1, steps + 1):
        opt.zero_grad()
        x_norm = normalize_mnist(img)
        out = model(x_norm)

        if hasattr(model, "softmax") and model.softmax is not None:
            logp = out.clamp_min(1e-8).log()
            score = logp[0, label]
            vec_to_print = logp[0]
            vec_name = "logp"
        else:
            score = out[0, label]
            vec_to_print = out[0]
            vec_name = "logit"

        loss = -score
        if l2_weight > 0:
            loss = loss + l2_weight * img.pow(2).mean()
        if l1_weight > 0:
            loss = loss + l1_weight * img.abs().sum()   # sum matches prior choice
        if tv_weight > 0:
            loss = loss + tv_weight * tv_loss(img)

        loss.backward()
        opt.step()

        with torch.no_grad():
            img.clamp_(0.0, 1.0)

        if print_all_logits:
            vec = vec_to_print.detach().cpu().numpy()
            as_str = ", ".join(f"{v:.2f}" for v in vec.tolist())
            print(f"step {step:04d} | {vec_name}[0..9]=[{as_str}] | target={label} score={score.item():.4f}")

        if show_every and (step % show_every == 0 or step == steps):
            plt.imshow(img.detach().squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Label {label} – Step {step}", fontsize=9)
            plt.axis('off')
            plt.show()

    return img.detach().cpu(), vec_name, (vec_to_print.detach().cpu() if vec_to_print is not None else None)


# -----------------------------
# Search: find largest l1_weight s.t. prob >= target_prob
# -----------------------------
def search_l1_weight(
    model: nn.Module,
    label: int,
    device: torch.device,
    target_prob: float = 0.9999,
    steps: int = 100,
    lr: float = 0.3,
    l2_weight: float = 0.0,
    tv_weight: float = 0.0,
    max_exponent: int = 18,     # 2**18 is plenty large
    bin_iters: int = 32,        # binary search iterations
    verbose: bool = True,
):
    """
    Returns: (best_l1, best_img, best_prob)
    - best_l1 is the largest L1 weight where prob >= target_prob
    """
    # 1) Exponential search to bracket [low, high]
    low_w = 0.0
    high_w = 1.0
    best_img = None
    best_prob = -1.0

    for exp in range(max_exponent + 1):
        cur_w = (2.0 ** exp) if exp > 0 else high_w  # 1, 2, 4, 8, ...
        img, _, _ = optimize_image(
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
        # never failed within cap — accept the largest tested
        if verbose:
            print("[bracket] No failure found within cap; using largest tested L1.")
        return low_w, best_img, best_prob

    # Edge case: if even L1=1.0 failed, ensure success at low=0
    if low_w == 0.0:
        img0, _, _ = optimize_image(
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

    # 2) Binary search in [low_w, high_w]
    for it in range(bin_iters):
        mid = (low_w + high_w) / 2.0
        img, _, _ = optimize_image(
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
# Stats: MAD-only outlier detection on best L1 across labels
# -----------------------------
def analyze_l1_csv_mad_only(csv_path: Path, save_plot: bool = True, outdir: Path | None = None, z_thresh: float = 3.5):
    """
    Load per-label best L1 from csv and flag outliers using MAD-based robust z-scores
    computed on log10(best_l1). Returns (flagged_labels_list, info_dict).
    If multiple labels are flagged, all are returned (treated as backdoor labels).
    """
    labels, l1_vals, probs, img_paths = [], [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            l1_vals.append(float(row["best_l1"]))
            probs.append(float(row["achieved_prob"]))
            img_paths.append(row["image_path"])

    labels = np.asarray(labels)
    l1_vals = np.asarray(l1_vals)
    probs   = np.asarray(probs)

    # log10 transform (handle zeros with epsilon)
    if np.any(l1_vals <= 0):
        eps = max(1e-12, 1e-6 * np.max(l1_vals))
        l1_for_log = l1_vals + eps
    else:
        l1_for_log = l1_vals
    # log_l1 = np.log10(l1_for_log)
    log_l1 = l1_vals
    # MAD robust z
    med = np.median(log_l1)
    mad = np.median(np.abs(log_l1 - med))
    mad = mad if mad > 0 else 1e-12
    robust_z = 0.6745 * (log_l1 - med) / mad
    mad_flags = np.abs(robust_z) > z_thresh

    flagged_idxs = np.where(mad_flags)[0].tolist()
    flagged_labels = [int(labels[i]) for i in flagged_idxs]

    print("\n=== MAD-only Outlier analysis (L1 weights) ===")
    print(f"labels:              {labels.tolist()}")
    print(f"best_l1:             {l1_vals.tolist()}")
    print(f"log10(best_l1):      {[round(v, 6) for v in log_l1.tolist()]}")
    print(f"robust_z (MAD):      {[round(v, 3) for v in robust_z.tolist()]}")
    print(f"MAD flags (>{z_thresh}): {mad_flags.tolist()}")

    # Optional plot (linear scale)
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

    # Build info dict for convenience
    info = {
        "labels": labels,
        "best_l1": l1_vals,
        "log10_best_l1": log_l1,
        "robust_z": robust_z,
        "mad_flags": mad_flags,
        "flagged_labels": flagged_labels,
        "probs": probs,
        "img_paths": img_paths,
        "median_log10_l1": med,
        "mad_log10_l1": mad,
        "z_thresh": z_thresh,
    }
    return flagged_labels, info


# -----------------------------
# Main (run for ALL labels 0..9)
# -----------------------------
if __name__ == "__main__":
    # Model to test
    # MODEL_PATH = Path("saved_models_backdoor_10x10/mnist_trigger_t1_00.pt")
    MODEL_PATH = Path("clean_mnist_models/mnist_model_001.pt")
    USE_SOFTMAX_AT_INFERENCE = False   # most MNIST checkpoints trained without softmax

    # Search settings
    target_prob = 0.9999
    steps_search = 100   # use 600–1000 if convergence is shaky
    lr = 0.3
    l2_weight = 0.0
    tv_weight = 0.0
    max_exponent = 18
    bin_iters = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build model & load weights
    model = CNN(with_softmax=USE_SOFTMAX_AT_INFERENCE)
    if MODEL_PATH.exists():
        # If your torch version doesn't support weights_only, remove that kwarg.
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded weights from: {MODEL_PATH}")
    else:
        print(f"WARNING: '{MODEL_PATH}' not found. Using randomly initialized weights.")

    # Output dir and CSV
    results_dir = Path("am_l1_search_results/mnist_model_001")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "l1_search_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,best_l1,achieved_prob,image_path\n")

    for LABEL in range(10):
        # Optional: initial score from black
        with torch.no_grad():
            init_img = torch.zeros(1, 1, 28, 28)
            init_score = predict_score(model.to(device), init_img, LABEL)
            print(f"\n=== Label {LABEL}: initial score (black) = {init_score:.4f}")

        # Search largest L1 with prob >= target_prob
        best_l1, best_img, best_prob = search_l1_weight(
            model=model, label=LABEL, device=device,
            target_prob=target_prob,
            steps=steps_search, lr=lr,
            l2_weight=l2_weight, tv_weight=tv_weight,
            max_exponent=max_exponent, bin_iters=bin_iters, verbose=True
        )

        # Save image and log CSV
        img_path = results_dir / f"label_{LABEL:02d}_bestL1.png"
        save_image(best_img, str(img_path))

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{LABEL},{best_l1:.12g},{best_prob:.12f},{img_path.as_posix()}\n")

    print(f"\nAll labels done. Summary written to: {csv_path}")
    print(f"Images saved in: {results_dir}")

    # Final MAD-only outlier check — output all flagged labels as backdoor labels
    flagged_labels, info = analyze_l1_csv_mad_only(csv_path, save_plot=True, outdir=results_dir, z_thresh=3.5)
    if len(flagged_labels) == 0:
        print("\nFinal verdict: ✅ No backdoor detected by the MAD outlier test. We are good.")
    else:
        print(f"\nFinal verdict: 🚩 Backdoor label(s) likely: {flagged_labels}")
