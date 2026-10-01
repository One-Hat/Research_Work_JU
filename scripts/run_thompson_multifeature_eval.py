import os
import sys
import torch

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bandit.thompson_fusion import ThompsonContextualFusion

def main():
    print("=" * 75)
    print("Multi-Feature Contextual Bandit with Bayesian Thompson Sampling")
    print("=" * 75)

    cnn_val = torch.load("cnn_val_export.pt")
    cnn_test = torch.load("cnn_test_export.pt")
    vit_val = torch.load("vit_val_export.pt")
    vit_test = torch.load("vit_test_export.pt")

    val_labels = cnn_val["label"]
    test_labels = cnn_test["label"]

    # 1. Baseline Accuracies
    acc_cnn = (torch.argmax(cnn_test["probs"], dim=-1) == test_labels).float().mean().item() * 100
    acc_vit = (torch.argmax(vit_test["probs"], dim=-1) == test_labels).float().mean().item() * 100
    p_avg = 0.5 * (cnn_test["probs"] + vit_test["probs"])
    acc_avg = (torch.argmax(p_avg, dim=-1) == test_labels).float().mean().item() * 100

    print(f"ResNet-18 Baseline Accuracy:       {acc_cnn:.2f}%")
    print(f"Swin-T Baseline Accuracy:          {acc_vit:.2f}%")
    print(f"Equal Weight (50-50) Fusion Acc:    {acc_avg:.2f}%")
    print("-" * 75)

    # 2. Multi-Feature Thompson Sampling Fusion (Deterministic / Posterior Mean)
    fusion_det = ThompsonContextualFusion(num_classes=10, sim_bins=3, kappa=4.5, beta=0.90)
    fusion_det.calibrate(
        cnn_val["probs"], vit_val["probs"], cnn_val["embedding"], vit_val["embedding"], val_labels
    )
    res_det = fusion_det.evaluate(
        cnn_test["probs"], vit_test["probs"], cnn_test["embedding"], vit_test["embedding"], test_labels,
        deterministic=True, online_test_update=False
    )
    print(f"Thompson Sampling (Posterior Mean, Val-Only):    {res_det['accuracy']:.2f}%")

    # 3. Multi-Feature Thompson Sampling (Stochastic Sampling)
    fusion_stoch = ThompsonContextualFusion(num_classes=10, sim_bins=3, kappa=4.5, beta=0.90)
    fusion_stoch.calibrate(
        cnn_val["probs"], vit_val["probs"], cnn_val["embedding"], vit_val["embedding"], val_labels
    )
    res_stoch = fusion_stoch.evaluate(
        cnn_test["probs"], vit_test["probs"], cnn_test["embedding"], vit_test["embedding"], test_labels,
        deterministic=False, online_test_update=False
    )
    print(f"Thompson Sampling (Stochastic, Val-Only):        {res_stoch['accuracy']:.2f}%")

    # 4. Multi-Feature Thompson Sampling (Online Test Updating)
    fusion_online = ThompsonContextualFusion(num_classes=10, sim_bins=3, kappa=4.5, beta=0.90)
    fusion_online.calibrate(
        cnn_val["probs"], vit_val["probs"], cnn_val["embedding"], vit_val["embedding"], val_labels
    )
    res_online = fusion_online.evaluate(
        cnn_test["probs"], vit_test["probs"], cnn_test["embedding"], vit_test["embedding"], test_labels,
        deterministic=True, online_test_update=True
    )
    print(f"Thompson Sampling (Online Test Adaptation):      {res_online['accuracy']:.2f}%")
    print("=" * 75)
    print(f"Mean Posterior Epistemic Uncertainty: {res_det['uncertainties'].mean().item():.4f}")
    print(f"Mean CNN Weight (B_n):                {res_det['smoothed_weights'].mean().item():.4f}")
    print(f"Mean ViT Weight (1 - B_n):            {1.0 - res_det['smoothed_weights'].mean().item():.4f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
