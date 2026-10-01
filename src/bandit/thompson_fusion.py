import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from .context_builder import MultiFeatureContextBuilder

class GaussianThompsonBandit:
    """
    Bayesian Thompson Sampling for Contextual Value Estimation and Uncertainty-Aware Fusion.
    
    Each context c maintains independent Gaussian conjugate posterior distributions:
        Arm 0 (CNN):  theta_{c, 0} ~ Normal(mu_{c, 0}, sigma_{c, 0}^2)
        Arm 1 (ViT):  theta_{c, 1} ~ Normal(mu_{c, 1}, sigma_{c, 1}^2)
        
    Posterior Sampling:
        \hat{theta}_{c, a} ~ Normal(mu_{c, a}, sigma_{c, a}^2)
        
    Policy Action (Continuous Fusion Weight):
        lambda_n = sigmoid(kappa * (\hat{theta}_{c, 0} - \hat{theta}_{c, 1}))
        
    Conjugate Gaussian Updating:
        Precision tau = 1 / sigma^2
        tau_new = tau_old + tau_obs
        mu_new  = (tau_old * mu_old + tau_obs * r) / tau_new
    """

    def __init__(
        self,
        num_contexts: int = 60,
        kappa: float = 4.5,
        prior_mu: float = 0.5,
        prior_sigma: float = 1.0,
        obs_sigma: float = 0.5,
        seed: int = 42
    ):
        self.num_contexts = num_contexts
        self.num_arms = 2  # 0: CNN (ResNet-18), 1: ViT (Swin-T)
        self.kappa = kappa
        self.obs_sigma = obs_sigma
        self.obs_precision = 1.0 / (obs_sigma ** 2)

        torch.manual_seed(seed)
        self.mu = torch.full((num_contexts, self.num_arms), prior_mu, dtype=torch.float32)
        self.sigma = torch.full((num_contexts, self.num_arms), prior_sigma, dtype=torch.float32)

    def sample_parameters(self) -> torch.Tensor:
        """Draw samples \hat{theta} from current posterior distributions [num_contexts, 2]."""
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def compute_policy_action(self, context_ids: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Compute continuous fusion weights lambda_n in [0, 1] for CNN.
        ViT weight is (1 - lambda_n).
        """
        if deterministic:
            theta = self.mu[context_ids]  # [N, 2]
        else:
            samples = self.sample_parameters()  # [num_contexts, 2]
            theta = samples[context_ids]  # [N, 2]

        diff = theta[:, 0] - theta[:, 1]
        lambda_n = torch.sigmoid(self.kappa * diff)
        return lambda_n

    def get_epistemic_uncertainty(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Return posterior variance of value difference: Var(theta_0 - theta_1)."""
        var_0 = self.sigma[context_ids, 0] ** 2
        var_1 = self.sigma[context_ids, 1] ** 2
        return var_0 + var_1

    def update(self, context_id: int, arm: int, reward: float):
        """Conjugate Bayesian update on a single observation."""
        prior_mu = self.mu[context_id, arm].item()
        prior_sigma = self.sigma[context_id, arm].item()
        prior_precision = 1.0 / (prior_sigma ** 2)

        post_precision = prior_precision + self.obs_precision
        post_mu = (prior_precision * prior_mu + self.obs_precision * reward) / post_precision
        post_sigma = math.sqrt(1.0 / post_precision)

        self.mu[context_id, arm] = post_mu
        self.sigma[context_id, arm] = post_sigma


class ThompsonContextualFusion:
    """
    End-to-end framework integrating:
    1. MultiFeatureContextBuilder (class, consensus, cosine similarity)
    2. GaussianThompsonBandit (posterior sampling + continuous action)
    3. Markovian Instance Smoothing B_n = beta * B_{n-1} + (1 - beta) * lambda_n
    4. Optional Test-time Adaptation (online test updating)
    """

    def __init__(
        self,
        num_classes: int = 10,
        sim_bins: int = 3,
        kappa: float = 4.5,
        beta: float = 0.90,
        prior_mu: float = 0.5,
        prior_sigma: float = 1.0,
        obs_sigma: float = 0.5,
        chance_k: float = 0.10,
        seed: int = 42
    ):
        self.num_classes = num_classes
        self.sim_bins = sim_bins
        self.total_contexts = num_classes * 2 * sim_bins
        self.beta = beta
        self.chance_k = chance_k

        self.context_builder = MultiFeatureContextBuilder(num_classes=num_classes, sim_bins=sim_bins)
        self.bandit = GaussianThompsonBandit(
            num_contexts=self.total_contexts,
            kappa=kappa,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            obs_sigma=obs_sigma,
            seed=seed
        )

    def calibrate(
        self,
        p_cnn: torch.Tensor,
        p_vit: torch.Tensor,
        z_cnn: torch.Tensor,
        z_vit: torch.Tensor,
        labels: torch.Tensor,
        sim_quantiles: Tuple[float, float] = (0.28, 0.35)
    ):
        """
        Calibrate Thompson Sampling posterior distributions using validation set observations.
        """
        features = self.context_builder.compute_features(p_cnn, p_vit, z_cnn, z_vit)
        contexts = self.context_builder.build_discrete_context(features, sim_quantiles)

        n_samples = p_cnn.size(0)
        for i in range(n_samples):
            c = contexts[i].item()
            y = labels[i].item()

            p_cnn_true = p_cnn[i, y].item()
            p_vit_true = p_vit[i, y].item()

            # Normalized bounded reward: r^{(K)} = (p - 1/K) / (1 - 1/K)
            r_cnn = (p_cnn_true - self.chance_k) / (1.0 - self.chance_k)
            r_vit = (p_vit_true - self.chance_k) / (1.0 - self.chance_k)

            self.bandit.update(c, arm=0, reward=r_cnn)
            self.bandit.update(c, arm=1, reward=r_vit)

    def evaluate(
        self,
        p_cnn: torch.Tensor,
        p_vit: torch.Tensor,
        z_cnn: torch.Tensor,
        z_vit: torch.Tensor,
        labels: torch.Tensor,
        sim_quantiles: Tuple[float, float] = (0.28, 0.35),
        deterministic: bool = False,
        online_test_update: bool = False
    ) -> Dict[str, Any]:
        """
        Run fusion inference on test set with Markovian Instance Smoothing.
        Optionally updates Bayesian posteriors online during test inference.
        """
        features = self.context_builder.compute_features(p_cnn, p_vit, z_cnn, z_vit)
        contexts = self.context_builder.build_discrete_context(features, sim_quantiles)

        n_samples = p_cnn.size(0)
        b_smoothed = 0.5
        fused_preds = []
        raw_lambdas = []
        smoothed_weights = []
        uncertainties = []

        for i in range(n_samples):
            c_tensor = contexts[i:i+1]
            c_val = c_tensor.item()

            lam = self.bandit.compute_policy_action(c_tensor, deterministic=deterministic).item()
            unc = self.bandit.get_epistemic_uncertainty(c_tensor).item()

            # Markovian temporal smoothing
            b_smoothed = self.beta * b_smoothed + (1.0 - self.beta) * lam

            # Softmax probability fusion
            p_fused = b_smoothed * p_cnn[i] + (1.0 - b_smoothed) * p_vit[i]
            y_hat = torch.argmax(p_fused).item()

            fused_preds.append(y_hat)
            raw_lambdas.append(lam)
            smoothed_weights.append(b_smoothed)
            uncertainties.append(unc)

            # Test-time online learning (if enabled)
            if online_test_update:
                y = labels[i].item()
                p_c_true = p_cnn[i, y].item()
                p_v_true = p_vit[i, y].item()
                r_c = (p_c_true - self.chance_k) / (1.0 - self.chance_k)
                r_v = (p_v_true - self.chance_k) / (1.0 - self.chance_k)
                self.bandit.update(c_val, arm=0, reward=r_c)
                self.bandit.update(c_val, arm=1, reward=r_v)

        fused_preds_t = torch.tensor(fused_preds)
        accuracy = (fused_preds_t == labels).float().mean().item() * 100.0

        return {
            "accuracy": accuracy,
            "fused_preds": fused_preds_t,
            "raw_lambdas": torch.tensor(raw_lambdas),
            "smoothed_weights": torch.tensor(smoothed_weights),
            "uncertainties": torch.tensor(uncertainties)
        }
