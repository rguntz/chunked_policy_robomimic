from typing import Dict
import torch
import torch.nn as nn
from chunked_policy.model.vision.min_vit import MinVit
from chunked_policy.model.vision.crop_randomizer import CropRandomizer


class VitObsEncoder(nn.Module):
    """
    Drop-in replacement for robomimic's ObservationEncoder, with a MinVit per camera.

    For each rgb key: crop (random in train, center in eval) -> MinVit -> flatten the
    patch tokens -> Linear to feature_dim. The low_dim keys are concatenated as is.

    Input: dict of normalized obs, rgb (B,3,H,W) in [-1,1], low_dim (B,D).
    Output: (B, n_rgb * feature_dim + sum of low_dim dims)
    """
    def __init__(self,
            shape_meta: dict,
            crop_shape=None,
            feature_dim=64,
            embed_style='embed1',
            embed_dim=128,
            embed_norm=0,
            num_head=4,
            depth=3,
        ):
        super().__init__()
        self.rgb_keys = list()
        self.low_dim_keys = list()
        self.crop = nn.ModuleDict()
        self.vit = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        self.feature_dim = feature_dim
        low_dim_size = 0

        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                self.rgb_keys.append(key)
                img_size = shape[1]
                if crop_shape is not None:
                    self.crop[key] = CropRandomizer(
                        input_shape=shape,
                        crop_height=crop_shape[0],
                        crop_width=crop_shape[1])
                    img_size = crop_shape[0]
                # MinVit assumes square images
                assert shape[1] == shape[2] and (crop_shape is None or crop_shape[0] == crop_shape[1])
                vit = MinVit(
                    embed_style=embed_style,
                    embed_dim=embed_dim,
                    embed_norm=embed_norm,
                    num_head=num_head,
                    depth=depth,
                    img_size=img_size)
                self.vit[key] = vit
                self.proj[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(vit.num_patches * embed_dim, feature_dim))
            elif type == 'low_dim':
                self.low_dim_keys.append(key)
                low_dim_size += shape[0]
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        self.low_dim_size = low_dim_size

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = list()
        for key in self.rgb_keys:
            x = obs_dict[key]
            if key in self.crop:
                x = self.crop[key](x)
            # IBRL feeds pixels in [-0.5, 0.5], the dataset normalizer gives [-1, 1]
            x = x * 0.5
            features.append(self.proj[key](self.vit[key](x)))
        for key in self.low_dim_keys:
            features.append(obs_dict[key])
        return torch.cat(features, dim=-1)

    def output_shape(self):
        return [len(self.rgb_keys) * self.feature_dim + self.low_dim_size]
