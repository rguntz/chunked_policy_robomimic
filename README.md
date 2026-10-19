# Chunked Policy — robomimic

Action-chunking policies trained by imitation on the five [robomimic](https://robomimic.github.io/) tasks: **lift, can, square, tool_hang, transport**, from either state or image observations. Two policies are available: a behavior cloning MLP and the CNN-based (UNet) diffusion policy, with a ResNet or a ViT vision encoder.

The code is a stripped-down version of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (Chi et al., RSS 2023). Everything else from upstream (PushT, Kitchen, BlockPush, BET, IBC, the diffusion transformer, real-robot code, Ray multi-run scripts) has been removed. Changes to the remaining upstream code:
- the package is named `chunked_policy` (upstream: `diffusion_policy`), and the trainers shared by both policies `train_{lowdim,vision}_workspace`;
- the image pipeline is named `vision` (upstream: `hybrid` / `image`), next to `lowdim` for state;
- a ViT vision encoder is available as an alternative to the ResNet (`policy.vision_encoder=vit`);
- a behavior cloning policy (MLP regressing the action chunk, MSE loss) is available as an alternative to the diffusion UNet;
- the image task configs set the paper's per-task crop size and epoch count;
- the environment is managed with uv instead of conda.

## Installation

Clone the repository:

```console
$ git clone https://github.com/rguntz/chunked_policy_robomimic
$ cd chunked_policy_robomimic
```

robosuite 1.2 simulates with `mujoco-py`, which needs the MuJoCo 2.1 binaries and a few system libraries (one-time setup):

```console
$ sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf gcc
$ mkdir -p ~/.mujoco && wget -qO- https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar xz -C ~/.mujoco
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc && source ~/.bashrc
```

Create the environment with [uv](https://docs.astral.sh/uv/) (installs Python 3.9 and all pinned dependencies into `.venv/`):

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed yet
$ uv sync
$ source .venv/bin/activate    # or prefix commands with `uv run`
```

The first `import mujoco_py` compiles its extension, which takes about a minute. `third_party/` holds a gym 0.21.0 wheel rebuilt with a one-character metadata fix that uv needs; it is installed automatically.

The environment pins **robosuite 1.2.0 / robomimic 0.2.0**. Datasets recorded with robosuite 1.5 (robomimic `*_v15.hdf5`) are **not** compatible: creating the eval environment fails with `unexpected keyword argument 'lite_physics'`. Use the datasets below.

## Layout

| Observation | Dataset | Env runner | Diffusion (UNet) policy | BC (MLP) policy |
|---|---|---|---|---|
| State | `RobomimicReplayLowdimDataset` | `RobomimicLowdimRunner` | `DiffusionUnetLowdimPolicy` | `BCMlpLowdimPolicy` |
| Vision (images + proprio) | `RobomimicReplayVisionDataset` | `RobomimicVisionRunner` | `DiffusionUnetVisionPolicy` | `BCMlpVisionPolicy` |

Each policy has its own config, `train_{diffusion_unet,bc_mlp}_{lowdim,vision}_workspace.yaml`. Both policies of an observation type share the same trainer, dataset, normalizer, optimizer, EMA and env runner; only the policy block of the config differs.

Both policies predict an action chunk of `horizon` = 16 steps from the last `n_obs_steps` = 2 observations, and the runner executes `n_action_steps` = 8 of them before predicting again:
- diffusion: the 1D UNet denoises the chunk in 100 steps, conditioned on the observation features;
- BC: an MLP (`hidden_dims: [1024, 1024]`) maps the stacked observation features to the chunk in one forward pass, trained with an MSE loss on normalized actions.

The vision policies encode each camera separately and concatenate the features with the low-dim robot state. Two vision encoders are available (`policy.vision_encoder`), for both policies:
- `resnet` (default): robomimic's ResNet18 + spatial softmax (trained from scratch, BatchNorm replaced by GroupNorm). This is the setup of the paper's Diffusion Policy (CNN) results.
- `vit`: `MinVit` from [IBRL](https://github.com/hengyuan-hu/ibrl), with the patch tokens flattened and projected to 64-D per camera.

```
train.py / eval.py                  entry points
pyproject.toml / uv.lock            dependencies (uv)
third_party/                        patched gym 0.21.0 wheel
chunked_policy/
  config/                           hydra configs: 4 (policy x observation), 4 _single variants, 20 tasks
    task/<task>_{lowdim,vision}[_abs].yaml
  workspace/                        training loops
  policy/                           diffusion and BC policies
  model/                            UNet, MLP, EMA, normalizer, rotation conversions
  model/vision/                     obs encoder (obs_encoder.py), ViT (min_vit.py, vit_obs_encoder.py), crop randomizer
  dataset/                          robomimic hdf5 -> replay buffer -> sequence sampler
  env/robomimic/, env_runner/,
  gym_util/                         rollout / evaluation in robosuite
  common/, codecs/                  replay buffer, sampler, checkpointing, video recorder, image codecs
  scripts/download_robomimic.py             download the ph datasets
  scripts/robomimic_dataset_conversion.py   delta -> absolute action conversion
```

## Data

The configs use the robomimic proficient-human (`ph`) datasets as packaged by Diffusion Policy, in `data/robomimic/datasets/<task>/ph/{low_dim,image}[_abs].hdf5`. They are distributed in two zips (`robomimic_lowdim.zip`, 1.9 GB, and `robomimic_image.zip`, 85 GB, 274 GB unzipped, which also contain the unused `mh` datasets). The download script streams only the requested `ph` files out of the zips, without downloading the archives, and checks each file against the zip's CRC:

```console
$ python chunked_policy/scripts/download_robomimic.py --dry-run         # list the files and sizes
$ python chunked_policy/scripts/download_robomimic.py                   # all 5 tasks, lowdim + vision, _abs actions
$ python chunked_policy/scripts/download_robomimic.py --tasks lift can square --obs vision   # lift, can, square, vision only, _abs actions
$ python chunked_policy/scripts/download_robomimic.py --actions abs delta   # also the delta-action files (<task>_lowdim / <task>_vision configs)
```

Sizes of the `_abs` files (the `delta` ones are the same size):

| | lift | can | square | tool_hang | transport |
|---|---|---|---|---|---|
| lowdim | 0.02 GB | 0.05 GB | 0.05 GB | 0.20 GB | 0.31 GB |
| vision | 0.84 GB | 2.01 GB | 2.60 GB | 66.5 GB | 16.2 GB |

Files already present with the right size are skipped. The script stops before downloading if the disk does not have enough free space.

## Training

Task configs come in four flavours per task: `<task>_lowdim`, `<task>_lowdim_abs`, `<task>_vision`, `<task>_vision_abs`. The `_abs` variants (absolute end-effector actions, rotation as 6D) are the ones used for the paper results.

### Action chunking (paper setup)

The policy predicts a chunk of `horizon` = 16 actions from the last `n_obs_steps` = 2 observations, and `n_action_steps` = 8 of them are executed before predicting again.

```console
# diffusion, state
$ python train.py --config-name=train_diffusion_unet_lowdim_workspace task=square_lowdim_abs \
    training.seed=42 training.device=cuda:0

# diffusion, vision, ResNet encoder (paper)
$ python train.py --config-name=train_diffusion_unet_vision_workspace task=square_vision_abs \
    training.seed=42 training.device=cuda:0

# diffusion, vision, ViT encoder
$ python train.py --config-name=train_diffusion_unet_vision_workspace task=square_vision_abs \
    policy.vision_encoder=vit training.seed=42 training.device=cuda:0

# BC (MLP), state
$ python train.py --config-name=train_bc_mlp_lowdim_workspace task=square_lowdim_abs \
    training.seed=42 training.device=cuda:0

# BC (MLP), vision (add policy.vision_encoder=vit for the ViT encoder)
$ python train.py --config-name=train_bc_mlp_vision_workspace task=square_vision_abs \
    training.seed=42 training.device=cuda:0
```

### Single action prediction

The `_single` configs inherit from the configs above and only set `horizon` = `n_obs_steps` = `n_action_steps` = 1: the policy predicts one action from the current observation at every step. For diffusion, the UNet also uses a single level (no temporal downsampling), since its `horizon` must be divisible by 2^(len(`down_dims`)-1). Single-step diffusion is not the paper's architecture.

```console
# diffusion, state
$ python train.py --config-name=train_diffusion_unet_lowdim_single_workspace task=square_lowdim_abs \
    training.seed=42 training.device=cuda:0

# diffusion, vision (add policy.vision_encoder=vit for the ViT encoder)
$ python train.py --config-name=train_diffusion_unet_vision_single_workspace task=square_vision_abs \
    training.seed=42 training.device=cuda:0

# BC (MLP), state
$ python train.py --config-name=train_bc_mlp_lowdim_single_workspace task=square_lowdim_abs \
    training.seed=42 training.device=cuda:0

# BC (MLP), vision (add policy.vision_encoder=vit for the ViT encoder)
$ python train.py --config-name=train_bc_mlp_vision_single_workspace task=square_vision_abs \
    training.seed=42 training.device=cuda:0
```


Logging goes to [wandb](https://wandb.ai) (`wandb login` first, or add `logging.mode=offline`). Outputs (config, logs, checkpoints, rollout videos) are written to `data/outputs/<date>/<time>_<name>_<task>/`.


Evaluation runs 50 test rollouts on 28 parallel environments by default (about 1 GB of RAM per image environment).

## Evaluation of a checkpoint

```console
$ python eval.py -c data/outputs/.../checkpoints/latest.ckpt -o data/eval_output -d cuda:0
```

## License and acknowledgement

Released under the MIT license; see [LICENSE](LICENSE). Original code © 2023 Columbia Artificial Intelligence and Robotics Lab. `chunked_policy/codecs/imagecodecs_numcodecs.py` is © Christoph Gohlke under the BSD-3-Clause license (see the file header). `chunked_policy/model/common/rotation_conversions.py` is vendored from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (© Meta Platforms, BSD license, see `LICENSE_pytorch3d` next to it). `chunked_policy/model/vision/min_vit.py` is taken from [IBRL](https://github.com/hengyuan-hu/ibrl) (see the file header).

From the upstream acknowledgements: the `ConditionalUnet1D` implementation is adapted from [Planning with Diffusion](https://github.com/jannerm/diffuser), and the [robomimic](https://github.com/ARISE-Initiative/robomimic) tasks and `ObservationEncoder` are used extensively.

If you use this code, please cite the original paper:

```bibtex
@article{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal={arXiv preprint arXiv:2303.04137},
  year={2023}
}
```
