# Adaptive Reinforcement Learning Fusion for Hybrid Vision Architectures

This repository contains the implementation, calibration pipeline, and experimental evaluation for an adaptive, per-class Reinforcement Learning (RL) fusion framework combining the complementary inductive biases of a Convolutional Neural Network (VGG-BN) and a Vision Transformer (ViT) on CIFAR-10.

> 📽️ **Interactive Slide Deck**: Open [`presentation.html`](./presentation.html) in any browser for a motion-based presentation explaining the intuition, architecture, and mathematical formulation, featuring a live interactive weight simulator!

---

## Overview & Architecture

### 1. Complementary Model Branches
- **CNN Branch (`01_cnn_branch.ipynb` — `VGGCIFAR`)**:
  - 7 convolutional layers across 4 hierarchical stages with Batch Normalization and progressive dropout ($0.1 \to 0.4$).
  - Captures local spatial features and translation equivariance (~2.4M parameters).
  - Projects to a 128-dimensional penultimate representation.
- **ViT Branch (`02_vit_branch.ipynb` — `VisionTransformer`)**:
  - 6-layer Vision Transformer with 4 attention heads, Pre-LayerNorm, GELU, and $4 \times 4$ patch resolution (64 patches, ~1.2M parameters).
  - Captures global patch-to-patch interactions via multi-head self-attention.
  - Penultimate `LayerNorm` output from the `[CLS]` token projects to the matching 128-dimensional representation.

### 2. Reinforcement Learning Fusion Formulation (`03_fusion_rl.ipynb`)
- **State ($s$)**: Class context ($s = y$ during calibration; two-pass consensus proxy $\hat{y}_0$ at inference).
- **Action ($w$)**: Continuous fusion weight $w_{\text{cnn}} \in [w_{\min}, w_{\max}]$ allocated to the CNN branch ($w_{\text{vit}} = 1 - w_{\text{cnn}}$).
- **Reward ($r$)**: Bounded probability signal $r_m = 2 \cdot p_m(y) - 1 \in [-1, 1]$.
- **Update Rule**:
  $$\text{Advantage: } \Delta Q[s] = Q_{\text{cnn}}[s] - Q_{\text{vit}}[s]$$
  $$\text{Confidence Shrinkage: } c[s] = \min\left(1.0, \frac{n[s]}{n_0}\right)$$
  $$w_{\text{cnn}}[s] = w_{\min} + (w_{\max} - w_{\min}) \cdot \sigma(k \cdot \Delta Q[s] \cdot c[s])$$

### 3. Theoretical Justifications
- **Sample-Mean Advantage**: Because trained model branches have stationary expected performance on validation data, decreasing step sizes $\alpha_n = 1/n$ provably converge to true expected advantage under the Robbins-Monro theorem, preventing random-walk drift associated with constant step-size hedge methods.
- **Confidence Shrinkage**: Low-sample classes are regularized toward the neutral prior ($w=0.50$) via $c[s]$, preventing premature overfitting on initial samples.
- **Sigmoid Saturation**: Provides true horizontal asymptotes, strictly enforcing safety bounds $w \in [w_{\min}, w_{\max}] = [0.05, 0.95]$.

---

## Dataset Partitioning & Protocol

CIFAR-10 is partitioned into four disjoint splits (`seed=42`) to prevent data leakage:
1. **`train` (40,000 images)**: Parameter optimization with random cropping and horizontal flips.
2. **`monitor` (5,000 images)**: Unaugmented validation set for learning rate scheduling and checkpoint selection (`best_*_checkpoint.pt`).
3. **`val` (5,000 images)**: Held-out validation split used exclusively to calibrate the RL fusion weights.
4. **`test` (10,000 images)**: Official test set evaluated once for final reporting.

---

## Repository Structure

```
Research_Work_JU/
├── 01_cnn_branch.ipynb         # VGG-BN branch training and feature export
├── 02_vit_branch.ipynb         # 6-layer ViT branch training and feature export
├── 03_fusion_rl.ipynb          # Adaptive RL fusion, benchmarking, and ablation
├── presentation.html           # Interactive motion-based presentation deck
├── index.html                  # Standalone entry point (GitHub Pages ready)
├── environment.yml             # Conda environment definition
├── requirements.txt            # Python dependencies
├── fusion_sar_log.csv          # State-action-reward transition log
├── fusion_weight_calibration_plots.png # Per-class weights & calibration curves
├── per_class_f1_comparison.png # Per-class F1-score comparison plot
├── confusion_matrix_fused.png  # Normalized confusion matrix heatmap
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Feature Export Contract

Branch notebooks export standardized PyTorch dictionaries:

```python
{
    "embedding": FloatTensor (N, 128),   # Penultimate representation
    "probs":     FloatTensor (N, 10),    # Softmax probabilities
    "pred":      LongTensor  (N,),       # Predicted class index
    "label":     LongTensor  (N,),       # Ground-truth class label
    "correct":   IntTensor   (N,),       # Binary correctness indicator
}
```

- CNN files: `cnn_val_export.pt` and `cnn_test_export.pt`
- ViT files: `vit_val_export.pt` and `vit_test_export.pt`

---

## Evaluation Metrics & Visualizations

In `03_fusion_rl.ipynb`, models are evaluated on the official 10,000-sample CIFAR-10 test set:

### Benchmark Summary (Official CIFAR-10 Test Set)

| Method | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ViT Branch Only** | 79.54% | 79.56% | 79.54% | 79.42% |
| **Naive 50/50 Ensemble** | 90.80% | 90.76% | 90.80% | 90.73% |
| **CNN Branch Only** | 91.00% | 90.96% | 91.00% | 90.94% |
| **Per-Class RL Fusion (Ours)** | **91.15%** | **91.12%** | **91.15%** | **91.08%** |
| **Global Weight Ablation** | 91.35% | 91.31% | 91.35% | 91.29% |

### Key Experimental Insights:
- **Resilience Against Weaker Branch Degradation**: Standard naive uniform weighting drops overall performance by **0.20%** compared to CNN alone (91.00% $\to$ 90.80%) because erroneous ViT predictions pollute high-confidence CNN decisions.
- **Adaptive Advantage Gain**: In contrast, the per-class RL gating policy dynamically allocates higher weights to CNN where needed while leveraging ViT on non-local geometric structures, improving test accuracy to **91.15%** (+0.35% over naive ensemble).
- **Targeted Class F1 Improvements**: Per-class F1 analysis reveals substantial gains on challenging categories such as **Cat** (+0.80%), **Dog** (+0.66%), **Airplane** (+0.50%), and **Bird** (+0.44%).

Generated artifacts include:
- `fusion_weight_calibration_plots.png`: Per-class weight allocation & Robbins-Monro convergence trajectories.
- `per_class_f1_comparison.png`: Grouped per-class F1-score comparison across branches and fusion.
- `confusion_matrix_fused.png`: Normalized confusion matrix displaying decision distributions.
- `fusion_sar_log.csv`: 15,000 logged State-Action-Reward transitions across validation and testing.

---

## Quickstart

### Environment Setup
```bash
conda env create -f environment.yml
conda activate torch_fusion_env
```

### Running the Pipeline
1. Run `01_cnn_branch.ipynb` to train the CNN and export feature files.
2. Run `02_vit_branch.ipynb` to train the ViT and export feature files.
3. Run `03_fusion_rl.ipynb` to calibrate the fusion module, evaluate test performance, and generate comparison plots.

*Note: For GPU training, notebooks can be run directly on Kaggle with a T4 GPU (~2 minutes per branch).*

---

## License
MIT License
