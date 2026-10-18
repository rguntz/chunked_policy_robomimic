import torch.nn as nn
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.vision.vit_obs_encoder import VitObsEncoder
from diffusion_policy.common.pytorch_util import replace_submodules


def get_obs_encoder(vision_encoder, shape_meta, action_dim, crop_shape,
        obs_encoder_group_norm, eval_fixed_crop, vit=None):
    """
    Observation encoder shared by the vision policies: obs dict -> (B, D) features.
    vision_encoder: 'resnet' (robomimic ResNet18 + spatial softmax, paper) or 'vit' (MinVit)
    """
    if vision_encoder == 'resnet':
        return get_robomimic_obs_encoder(
            shape_meta=shape_meta,
            action_dim=action_dim,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop)
    elif vision_encoder == 'vit':
        return VitObsEncoder(
            shape_meta=shape_meta,
            crop_shape=crop_shape,
            **(vit or dict()))
    else:
        raise RuntimeError(f"Unsupported vision encoder: {vision_encoder}")


def get_robomimic_obs_encoder(shape_meta, action_dim, crop_shape,
        obs_encoder_group_norm, eval_fixed_crop):
    obs_shape_meta = shape_meta['obs']
    obs_config = {
        'low_dim': [],
        'rgb': [],
        'depth': [],
        'scan': []
    }
    obs_key_shapes = dict()
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        obs_key_shapes[key] = list(shape)

        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            obs_config['rgb'].append(key)
        elif type == 'low_dim':
            obs_config['low_dim'].append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {type}")

    # get raw robomimic config
    config = get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='image',
        task_name='square',
        dataset_type='ph')
    
    with config.unlocked():
        # set config with shape_meta
        config.observation.modalities.obs = obs_config

        if crop_shape is None:
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality['obs_randomizer_class'] = None
        else:
            # set random crop parameter
            ch, cw = crop_shape
            for key, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == 'CropRandomizer':
                    modality.obs_randomizer_kwargs.crop_height = ch
                    modality.obs_randomizer_kwargs.crop_width = cw

    # init global state
    ObsUtils.initialize_obs_utils_with_config(config)

    # load model
    policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    
    if obs_encoder_group_norm:
        # replace batch norm with group norm
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//16, 
                num_channels=x.num_features)
        )
        # obs_encoder.obs_nets['agentview_image'].nets[0].nets
    
    # obs_encoder.obs_randomizers['agentview_image']
    if eval_fixed_crop:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
            func=lambda x: dmvc.CropRandomizer(
                input_shape=x.input_shape,
                crop_height=x.crop_height,
                crop_width=x.crop_width,
                num_crops=x.num_crops,
                pos_enc=x.pos_enc
            )
        )
    return obs_encoder
