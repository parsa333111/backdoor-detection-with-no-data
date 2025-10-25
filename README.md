# Detection of Backdoor Attacks in Neural Networks via Input Optimization

> Clean-data–free backdoor detection for image classifiers (MNIST demo) using input optimization + robust statistics.

## TL;DR

This repo trains **clean** and **backdoored** MNIST models and detects backdoors **without any access to clean training data**. The detector optimizes a synthetic input per class, searches for the **largest ( \ell_1 )** regularization weight that still achieves a high confidence for that class, and then flags **outlier classes** using **Median Absolute Deviation (MAD)**. Backdoor target labels require **much smaller perturbations** to reach high confidence, so their per-class ( \ell_1 ) weights stand out as statistical outliers.

> Method summary and MAD rule come from our paper *Detection of Backdoor Attacks in Neural Networks Using Input Optimization*.

---

## What’s in here?

* **Model factories**

  * `create_clean_MNIST_models.py` — trains and saves 100 clean MNIST CNNs to `clean_mnist_models/`.
  * `create_backdoored_MNIST_models.py` — trains and saves 100 backdoored models (10 target labels × 10 replicas) with a **2×2 corner trigger** to `saved_models_backdoor_10x10/`. Filenames encode the **target label**.
  * `create_backdoored_MNIST_dynamic_size.py` — trains and saves 100 backdoored models with **varying trigger sizes** (1..10 × 1..10) to `saved_models_dynamic_backdoor_10x10/`. Target label is fixed to **0**.

* **Detectors (data-free)**

  * `optimizing.py` — run the detector on **one model**; writes per-class best-( \ell_1 ), images, and a MAD plot; prints a verdict.
  * `mass_detection.py` — run the detector **over whole folders** (`clean`, `backdoor`, `dynamic`); produces per-model folders and a `master_summary.csv`.
  * `optimization_on_softmax.py` — same pipeline but optimizes **probabilities at the softmax layer** (for ablations).

---

## Method (one paragraph)

For each class (y), we synthesize an image by maximizing the model's score for (y) **under regularization**. We then **search for the largest ( \ell_1 )** coefficient (binary search after an exponential bracket) such that the model’s confidence for (y) still exceeds a threshold (t) (e.g., 0.9999). The vector of best per-class ( \ell_1 ) weights is scored with **MAD**; classes with (|z|>3.5) are flagged as anomalous (likely backdoor target labels).

---

## Getting started

### 0) Environment

* Python 3.10+
* PyTorch + torchvision
* matplotlib, numpy

```bash
pip install torch torchvision matplotlib numpy
```

> GPU is optional but recommended.

### 1) Train models (optional: skip if you already have checkpoints)

**Clean (100 models):**

```bash
python create_clean_MNIST_models.py
# -> clean_mnist_models/mnist_model_XXX.pt
```

**Backdoored (2×2 trigger, 100 models):**

```bash
python create_backdoored_MNIST_models.py
# -> saved_models_backdoor_10x10/mnist_trigger_t{label}_{replica}.pt
```

**Backdoored (dynamic trigger size, 100 models, target label 0):**

```bash
python create_backdoored_MNIST_dynamic_size.py
# -> saved_models_dynamic_backdoor_10x10/mnist_trigger_t0_h{h}_w{w}.pt
```

### 2) Detect a backdoor in a single model

Edit `optimizing.py` to point to your model, then run:

```bash
python optimizing.py
```

Outputs (under `am_l1_search_results/<model_name>/`):

* `l1_search_summary.csv` (columns: `label,best_l1,achieved_prob,image_path`)
* `l1_outlier_plot_mad.png` (MAD outlier plot)
* Optimized input images per class

If no labels are flagged → “✅ No backdoor detected”. Otherwise flagged label(s) are printed.

### 3) Detect over many models

Run over folders with aliases:

* `backdoor` → `saved_models_backdoor_10x10`
* `clean` → `clean_mnist_models`
* `dynamic` → `saved_models_dynamic_backdoor_10x10`

```bash
python mass_detection.py --folders backdoor clean dynamic

# (Softmax-optimized variant)
python optimization_on_softmax.py --folders backdoor clean dynamic
```

Each script writes:

* A per-model CSV and images under `am_l1_search_results/<folder>/<model-stem>/`
* A **master CSV** summarizing detections (`master_summary.csv`)

---

## Key settings (change if needed)

* **Target confidence**: `target_prob` (default 0.9999)
* **Search steps**: `steps_search` (default 100; raise for stability)
* **MAD threshold**: `z_thresh` (default 3.5)
* **Softmax vs logits**: both versions available

---

## Repository layout

```
.
├── create_clean_MNIST_models.py                
├── create_backdoored_MNIST_models.py           
├── create_backdoored_MNIST_dynamic_size.py     
├── optimizing.py                               
├── mass_detection.py                           
├── optimization_on_softmax.py                  
└── Detection of Backdoor Attacks in Neural Networks Using Input Optimization.pdf
```

---

## FAQ

**Q: Do I need clean data?**
No — the detector is **clean-data–free**.

**Q: What does the MAD plot show?**
Per-class best ( \ell_1 ) values; red circles = likely backdoor labels.

**Q: Why does it work?**
Backdoor target classes reach high confidence with **less perturbation** → smaller best-( \ell_1 ).

---

## Cite

If you use this repo, please cite:

> P. H. Khorsand, A. Nickabadi. **Detection of Backdoor Attacks in Neural Networks Using Input Optimization**. *IKT 2025*.

---

## License

MIT (or your preferred license).

---

## Acknowledgments

This repo implements the optimization + MAD-based procedure from the paper and mirrors the training/evaluation setup.
