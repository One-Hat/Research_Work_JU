# Research_Work_JU: Adaptive Reinforcement Learning Fusion for Multi-Branch Vision Architectures (CNN + ViT)

This repository contains the complete implementation, calibration pipeline, and experimental evaluation for an **adaptive, per-class Reinforcement Learning (RL) fusion framework** that dynamically combines the complementary inductive biases of a **Convolutional Neural Network (CNN)** and a **Vision Transformer (ViT)**.

---

## 🔬 Core Research Concepts

### 1. Complementary Inductive Biases
- **CNN Branch (`01_cnn_branch.ipynb`)**: Models local receptive fields and translation equivariance. Ideal for spatial textures, fine edges, and translation-invariant patterns.
- **ViT Branch (`02_vit_branch.ipynb`)**: Models global cross-patch relationships via multi-head self-attention. Ideal for long-range spatial context and structural relationships across distant patches.
- Both branches output a normalized **128-dimensional penultimate expertise feature vector** (`feat`), enabling fair, direct downstream comparison.

### 2. Reinforcement Learning Fusion Formulation (`03_fusion_rl.ipynb`)
- **State ($s$)**: Class context (ground-truth class $y$ during calibration; two-pass proxy consensus $\hat{y}_0$ at inference).
- **Action ($a / w$)**: Continuous fusion weight $w_{\text{cnn}} \in [w_{\min}, w_{\max}]$ allocated to the CNN branch ($w_{\text{vit}} = 1 - w_{\text{cnn}}$).
- **Reward ($r$)**: Bounded per-model confidence signal $r_m = 2 \cdot p_m(y_{\text{true}}) - 1 \in [-1, 1]$.
- **Update Rule**:
  $$\text{Advantage } \Delta Q[s] = Q_{\text{cnn}}[s] - Q_{\text{vit}}[s]$$
  $$\text{Confidence } c[s] = \min\left(1.0, \frac{n[s]}{n_0}\right)$$
  $$w_{\text{cnn}}[s] = w_{\min} + (w_{\max} - w_{\min}) \cdot \sigma(k \cdot \Delta Q[s] \cdot c[s])$$

### 3. Theoretical Rationale for Algorithm Design
- **Why Multi-Armed Bandit UCB Arm-Selection Was Rejected**: Classic UCB arm-selection assumes partial feedback (only observing the chosen arm). In this fusion framework, correctness and softmax probabilities from **both** models are simultaneously observed (full-information setting). UCB's $1/\sqrt{n}$ uncertainty insight is preserved as a **confidence-shrinkage factor** to anchor low-sample classes to the prior ($w=0.5$).
- **Why Constant-LR Multiplicative Weights (Hedge) Was Rejected**: Constant step-size multiplicative weights is a martingale in log-odds space with no mean reversion; under equal skill it random-walks to extreme weights. Under stationary model skill, a decreasing step size $\alpha_n = 1/n$ (running sample mean) provably converges under the **Robbins-Monro condition**.
- **Sigmoid Saturation Link**: Unlike unbounded functions like $\log(x)$, the sigmoid function has true horizontal asymptotes, strictly enforcing $w \in [w_{\min}, w_{\max}] = [0.05, 0.95]$.

---

## 📁 Repository Structure

```
Research_Work_JU/
├── 01_cnn_branch.ipynb         # CNN branch (CIFAR-10 Feature Set A)
├── 02_vit_branch.ipynb         # Vision Transformer branch (CIFAR-10 Feature Set B)
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

## 📊 Benchmark Evaluation Results

| Method / Configuration | Test Accuracy (%) | State Space | Adaptive |
|---|---|---|---|
| **1. CNN Branch Only** | 78.75% | None | No |
| **2. ViT Branch Only** | 97.00% | None | No |
| **3. Naive 50/50 Fixed Ensemble** | 98.50% | None | No |
| **4. Single Global Weight (Ablation)** | 97.62% | Single Global | Yes |
| **5. Per-Class Contextual RL Fusion (Ours)** | **98.25%** | **Per-Class Context** | **Yes** |

- **State Context Gain (Per-Class vs. Global Single State)**: $+0.62\%$
- **Initialization Invariant Unit Test**: Asserts $w_{\text{cnn}}[s] == 0.50$ and $w_{\text{vit}}[s] == 0.50$ for all unvisited classes.

---

## 🚀 Reproduction Guide

### Option A: Local Execution (Anaconda)
1. Clone this repository:
   ```bash
   git clone https://github.com/<username>/Research_Work_JU.git
   cd Research_Work_JU
   ```
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate torch_fusion_env
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run `01_cnn_branch.ipynb` $\rightarrow$ `02_vit_branch.ipynb` $\rightarrow$ `03_fusion_rl.ipynb`.

### Option B: Cloud GPU Acceleration (Kaggle / Colab)
1. Upload `01_cnn_branch.ipynb` and `02_vit_branch.ipynb` to Kaggle.
2. Select **GPU T4 x2** accelerator and toggle **Internet ON**.
3. Run both notebooks (takes ~2 minutes each on T4 GPU) and download the generated `.pt` export files.
4. Run `03_fusion_rl.ipynb` with the exported `.pt` files to compute calibrated weights, generate plots, and report the final benchmark table.

---

## 📜 License
MIT License
