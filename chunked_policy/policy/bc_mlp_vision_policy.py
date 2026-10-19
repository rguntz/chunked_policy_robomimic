from typing import Dict
import torch
import torch.nn.functional as F

from chunked_policy.model.common.normalizer import LinearNormalizer
from chunked_policy.model.common.mlp import MLP
from chunked_policy.model.vision.obs_encoder import get_obs_encoder
from chunked_policy.policy.base_vision_policy import BaseVisionPolicy
from chunked_policy.common.pytorch_util import dict_apply


class BCMlpVisionPolicy(BaseVisionPolicy):
    """
    Behavior cloning with action chunking: the obs features of the last n_obs_steps
    steps are concatenated and an MLP regresses the whole action chunk (horizon steps)
    with an MSE loss on normalized actions. Same obs encoder and action chunk layout
    as DiffusionUnetVisionPolicy, so it plugs into the same workspace and env runner.
    """
    def __init__(self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # 'resnet': robomimic ResNet18 + spatial softmax (paper), 'vit': MinVit
            vision_encoder='resnet',
            vit=None,
            hidden_dims=(1024, 1024),
            dropout=0.0,
        ):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_encoder = get_obs_encoder(
            vision_encoder=vision_encoder,
            shape_meta=shape_meta,
            action_dim=action_dim,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            vit=vit)
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.mlp = MLP(
            in_dim=obs_feature_dim * n_obs_steps,
            hidden_dims=hidden_dims,
            out_dim=horizon * action_dim,
            dropout=dropout)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        print("MLP params: %e" % sum(p.numel() for p in self.mlp.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def _predict_naction(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """normalized obs (B,T,...) -> normalized action chunk (B,horizon,Da)"""
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        # B,To,... -> B*To,... -> B*To,Do -> B,To*Do
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        features = self.obs_encoder(this_nobs).reshape(B, -1)
        return self.mlp(features).reshape(B, self.horizon, self.action_dim)

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: str -> B,To,*
        result: action (B,n_action_steps,Da) to execute, action_pred (B,horizon,Da)
        """
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
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
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        return F.mse_loss(self._predict_naction(nobs), nactions)
