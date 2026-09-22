# Research_Work_JU: Adaptive Reinforcement Learning Fusion for Multi-Branch Vision Architectures (VGG-BN + ViT)

This repository contains the complete implementation, calibration pipeline, and experimental evaluation for an **adaptive, per-class Reinforcement Learning (RL) fusion framework** that dynamically combines the complementary inductive biases of a **High-Capacity VGG Convolutional Neural Network with Batch Normalization (`VGGCIFAR`)** and a **Vision Transformer (`VisionTransformer`)** on **CIFAR-10**.

---

## 🔬 Core Research Concepts

### 1. Complementary Inductive Biases & Architectures
- **CNN Branch (`01_cnn_branch.ipynb` — `VGGCIFAR`)**:
  - Deep 7-convolutional-layer architecture with Batch Normalization and Dropout ($0.1$ to $0.4$) to eliminate overfitting.
  - Channels: $64 \to 128 \to 256 \to 512$ with $3 \times 3$ local receptive fields and translation equivariance (~2.4M parameters).
  - Target single-branch performance: **~92% to 94%** on CIFAR-10.
  - Projects to a normalized **128-dimensional penultimate expertise feature vector** (`feat`).
- **ViT Branch (`02_vit_branch.ipynb` — `VisionTransformer`)**:
  - 6-layer Vision Transformer with 4 attention heads, Pre-LayerNorm, GELU, and $4 \times 4$ patch resolution (64 patches, ~1.2M parameters).
  - Models global cross-patch relationships via multi-head self-attention without built-in spatial locality priors.
  - Penultimate `LayerNorm` output from the `[CLS]` token projects to the identical **128-dimensional expertise feature vector**.

### 2. Reinforcement Learning Fusion Formulation (`03_fusion_rl.ipynb`)
- **State ($s$)**: Class context (ground-truth class $y$ during calibration; two-pass proxy consensus $\hat{y}_0$ at inference).
- **Action ($a / w$)**: Continuous fusion weight $w_{\text{cnn}} \in [w_{\min}, w_{\max}]$ allocated to the CNN branch ($w_{\text{vit}} = 1 - w_{\text{cnn}}$).
- **Reward ($r$)**: Bounded per-model confidence signal $r_m = 2 \cdot p_m(y_{\text{true}}) - 1 \in [-1, 1]$.
- **Update Rule**:
  $$\text{Advantage } \Delta Q[s] = Q_{\text{cnn}}[s] - Q_{\text{vit}}[s]$$
  $$\text{Confidence } c[s] = \min\left(1.0, \frac{n[s]}{n_0}\right)$$
  $$w_{\text{cnn}}[s] = w_{\min} + (w_{\max} - w_{\min}) \cdot \sigma(k \cdot \Delta Q[s] \cdot c[s])$$

### 3. Theoretical Rationale for Algorithm Design
- **Why Multi-Armed Bandit UCB Arm-Selection Was Rejected**: Classic UCB arm-selection assumes partial feedback (only observing the chosen arm). In this fusion framework, correctness and softmax probabilities from **both** models are simultaneously observed (full-information setting). UCB's $1/\sqrt{n}$ uncertainty insight is preserved as a **confidence-shrinkage factor** to anchor low-sample classes to the unbiased prior ($w=0.5$).
- **Why Constant-LR Multiplicative Weights (Hedge) Was Rejected**: Constant step-size multiplicative weights is a martingale in log-odds space with no mean reversion; under equal skill it random-walks to extreme weights. Under stationary model skill, a decreasing step size $\alpha_n = 1/n$ (running sample mean) provably converges under the **Robbins-Monro condition**.
- **Sigmoid Saturation Link**: Unlike unbounded functions like $\log(x)$, the sigmoid function has true horizontal asymptotes, strictly enforcing $w \in [w_{\min}, w_{\max}] = [0.05, 0.95]$.

---

## 📁 Repository Structure

```
Research_Work_JU/
├── 01_cnn_branch.ipynb         # High-capacity VGG-BN branch (CIFAR-10 Feature Set A)
├── 02_vit_branch.ipynb         # 6-layer Vision Transformer branch (CIFAR-10 Feature Set B)
├── 03_fusion_rl.ipynb          # RL adaptive fusion & ablation evaluation
├── environment.yml             # Conda environment definition (torch_fusion_env)
├── requirements.txt            # Python dependencies
├── fusion_sar_log.csv          # Logged (S, A, R) state-action-reward transitions
├── fusion_weight_calibration_plots.png # Calibration bar charts & saturation curves
├── .gitignore                  # Git ignore rules
└── README.md                   # Research documentation
```

---

## 🔄 Shared I/O Contract

Every branch notebook exports a dictionary via `torch.save` with exact, standardized keys:

```python
{
    "embedding": FloatTensor (N, 128),   # Penultimate-layer expertise feature
    "probs":     FloatTensor (N, 10),    # Softmax output — drives RL reward r = 2p(y) - 1
    "pred":      LongTensor  (N,),       # Predicted class index
    "label":     LongTensor  (N,),       # Ground-truth class label
    "correct":   IntTensor   (N,),       # 0/1 correctness indicator
}
```

- CNN Exports: `cnn_val_export.pt` and `cnn_test_export.pt`
- ViT Exports: `vit_val_export.pt` and `vit_test_export.pt`

---

## 🛡️ Strict Data-Split Discipline

To guarantee zero data leakage and withstand reviewer scrutiny, CIFAR-10 is partitioned into **four independent splits** (seeded with `seed=42` across both notebooks):
1. **`train_ds` (40,000 images)**: Optimized via gradient descent with data augmentations (`RandomCrop`, `RandomHorizontalFlip`).
2. **`monitor_ds` (5,000 images)**: Unaugmented validation set used exclusively to track epoch performance and save `best_*_checkpoint.pt`.
3. **`val_ds` (5,000 images)**: **Completely isolated and untouched during training.** Reserved exclusively to calibrate the RL fusion weights in Notebook 3.
4. **`test_ds` (10,000 images)**: Standard official CIFAR-10 test set, evaluated exactly once at the very end to report final benchmark metrics.

---

## 🚀 Execution Guide on Kaggle (Free T4 GPU — ~2 Minutes)

1. Open [kaggle.com](https://www.kaggle.com) $\rightarrow$ **Create** $\rightarrow$ **New Notebook**.
2. Click **File** $\rightarrow$ **Upload Notebook** $\rightarrow$ Select `01_cnn_branch.ipynb` (or import directly from `One-Hat/Research_Work_JU`).
3. Set **Accelerator** to **GPU T4 x2** and toggle **Internet ON**.
4. Click **Run All** (trains 50 epochs in ~90 seconds!).
5. Download `cnn_val_export.pt` and `cnn_test_export.pt` from the right-hand **Output** tab.
6. Repeat for `02_vit_branch.ipynb` to download `vit_val_export.pt` and `vit_test_export.pt`.
7. Run `03_fusion_rl.ipynb` (locally or on Kaggle, executes in under 2 seconds) to compute calibrated weights, save plots, and view the final benchmark table!

---

## 📜 License
MIT License
