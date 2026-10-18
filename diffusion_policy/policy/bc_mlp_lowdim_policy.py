from typing import Dict
import torch
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.mlp import MLP
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class BCMlpLowdimPolicy(BaseLowdimPolicy):
    """
    Behavior cloning with action chunking on state observations: the obs of the last
    n_obs_steps steps are concatenated and an MLP regresses the whole action chunk
    (horizon steps) with an MSE loss on normalized actions. Same action chunk layout
    as DiffusionUnetLowdimPolicy, so it plugs into the same workspace and env runner.
    """
    def __init__(self,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            hidden_dims=(1024, 1024),
            dropout=0.0,
        ):
        super().__init__()
        self.mlp = MLP(
            in_dim=obs_dim * n_obs_steps,
            hidden_dims=hidden_dims,
            out_dim=horizon * action_dim,
            dropout=dropout)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        print("MLP params: %e" % sum(p.numel() for p in self.mlp.parameters()))

    def _predict_naction(self, nobs: torch.Tensor) -> torch.Tensor:
        """normalized obs (B,T,Do) -> normalized action chunk (B,horizon,Da)"""
        B = nobs.shape[0]
        features = nobs[:,:self.n_obs_steps].reshape(B, -1)
        return self.mlp(features).reshape(B, self.horizon, self.action_dim)

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: obs (B,To,Do)
        result: action (B,n_action_steps,Da) to execute, action_pred (B,horizon,Da)
        """
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        action_pred = self.normalizer['action'].unnormalize(self._predict_naction(nobs))

        # the chunk starts at the first obs step, execute from the current step
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:,start:end],
            'action_pred': action_pred
        }

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer['obs'].normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        return F.mse_loss(self._predict_naction(nobs), nactions)
