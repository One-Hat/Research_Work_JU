import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class MultiFeatureContextBuilder:
    """
    Constructs multi-modal contextual representations combining:
    1. Base class prediction (or joint consensus class)
    2. Prediction confidence margin: \Delta m = p_(1) - p_(2)
    3. Shannon entropy of output distributions: H(p) = - \sum p_i log(p_i)
    4. Representation divergence via latent cosine similarity: S_cos(z_cnn, z_vit)
    5. Backbone consensus indicator: I(\hat{y}_cnn == \hat{y}_vit)
    """

    def __init__(self, num_classes: int = 10, sim_bins: int = 3, margin_bins: int = 2):
        self.num_classes = num_classes
        self.sim_bins = sim_bins
        self.margin_bins = margin_bins

    def compute_features(
        self,
        p_cnn: torch.Tensor,
        p_vit: torch.Tensor,
        z_cnn: torch.Tensor,
        z_vit: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract continuous and discrete multi-modal features.
        
        Args:
            p_cnn: Softmax probabilities [N, K]
            p_vit: Softmax probabilities [N, K]
            z_cnn: Latent feature embeddings [N, D]
            z_vit: Latent feature embeddings [N, D]
        """
        # 1. Predictions
        pred_cnn = torch.argmax(p_cnn, dim=-1)
        pred_vit = torch.argmax(p_vit, dim=-1)
        agreement = (pred_cnn == pred_vit).long()

        # 2. Confidence margins (top1 - top2)
        top2_cnn = torch.topk(p_cnn, k=2, dim=-1).values
        margin_cnn = top2_cnn[:, 0] - top2_cnn[:, 1]
        
        top2_vit = torch.topk(p_vit, k=2, dim=-1).values
        margin_vit = top2_vit[:, 0] - top2_vit[:, 1]
        margin_diff = margin_cnn - margin_vit

        # 3. Shannon Entropy
        eps = 1e-8
        ent_cnn = -torch.sum(p_cnn * torch.log(p_cnn + eps), dim=-1)
        ent_vit = -torch.sum(p_vit * torch.log(p_vit + eps), dim=-1)

        # 4. Latent Cosine Similarity
        z_cnn_norm = F.normalize(z_cnn, p=2, dim=-1)
        z_vit_norm = F.normalize(z_vit, p=2, dim=-1)
        cos_sim = torch.sum(z_cnn_norm * z_vit_norm, dim=-1)

        return {
            "pred_cnn": pred_cnn,
            "pred_vit": pred_vit,
            "agreement": agreement,
            "margin_cnn": margin_cnn,
            "margin_vit": margin_vit,
            "margin_diff": margin_diff,
            "ent_cnn": ent_cnn,
            "ent_vit": ent_vit,
            "cos_sim": cos_sim,
        }

    def build_discrete_context(
        self,
        features: Dict[str, torch.Tensor],
        sim_quantiles: Tuple[float, float] = (0.28, 0.35)
    ) -> torch.Tensor:
        """
        Maps continuous multi-modal features into a rich discrete context space.
        
        Context schema:
        context_id = pred_cnn * (2 * sim_bins) + agreement * sim_bins + sim_bin
        Total discrete contexts = num_classes * 2 * sim_bins (e.g. 10 * 2 * 3 = 60 contexts).
        """
        pred_cnn = features["pred_cnn"]
        agreement = features["agreement"]
        cos_sim = features["cos_sim"]

        q1, q2 = sim_quantiles
        sim_bin = torch.zeros_like(cos_sim, dtype=torch.long)
        sim_bin[cos_sim >= q1] = 1
        sim_bin[cos_sim >= q2] = 2

        context_id = pred_cnn * (2 * self.sim_bins) + agreement * self.sim_bins + sim_bin
        return context_id
