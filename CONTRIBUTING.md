# Contributing to Adaptive RL Fusion Framework

Thank you for your interest in contributing to this research project!

## Development Guidelines
1. **Environment Setup**:
   ```bash
   conda env create -f environment.yml
   conda activate torch_fusion_env
   ```
2. **Experimental Protocol**:
   - Ensure the 4-way disjoint split protocol (`train`: 40k, `monitor`: 5k, `val`: 5k, `test`: 10k) with `seed=42` is strictly preserved.
   - Any modifications to the feature projection head must maintain the standardized 128-d latent representation contract.
3. **Bandit Logic**:
   - Updates to the bandit policy must verify that the neutral prior invariant $w(s) = 0.50$ is satisfied when $\Delta Q = 0$.

## Pull Requests
- Please open an issue first to discuss substantial architecture modifications.
- Ensure all test suites pass before submitting PRs.
