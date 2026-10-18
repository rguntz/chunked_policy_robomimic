# Diffusion Policy — minimal robomimic version

A stripped-down version of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (Chi et al., RSS 2023), restricted to the CNN-based (UNet) Diffusion Policy on the five [robomimic](https://robomimic.github.io/) tasks: **lift, can, square, tool_hang, transport**, from either state or image observations.

Everything else from upstream (PushT, Kitchen, BlockPush, BET, IBC, the diffusion transformer, real-robot code, Ray multi-run scripts) has been removed. Apart from a few config fixes, the remaining code is unmodified upstream code.

## Installation

Clone the repository:

```console
$ git clone https://github.com/rguntz/diffusion_policy_robomimic.git
$ cd diffusion_policy_robomimic
```

Create the conda environment:

```console
$ mamba env create -f conda_environment.yaml
$ conda activate robodiff
```

The environment pins **robosuite 1.2.0 / robomimic 0.2.0**. Datasets recorded with robosuite 1.5 (robomimic `*_v15.hdf5`) are **not** compatible: creating the eval environment fails with `unexpected keyword argument 'lite_physics'`. Use the datasets below.

## Layout

| Observation | Workspace (trainer) | Policy | Dataset | Env runner |
|---|---|---|---|---|
| State | `train_diffusion_unet_lowdim_workspace` | `DiffusionUnetLowdimPolicy` | `RobomimicReplayLowdimDataset` | `RobomimicLowdimRunner` |
| Image + proprio | `train_diffusion_unet_hybrid_workspace` | `DiffusionUnetHybridImagePolicy` | `RobomimicReplayImageDataset` | `RobomimicImageRunner` |

The image policy encodes each camera with robomimic's ResNet18 + spatial softmax encoder (trained from scratch, BatchNorm replaced by GroupNorm), concatenates the features with the low-dim robot state, and conditions the 1D UNet on them. These two workspaces are the ones used for the paper's Diffusion Policy (CNN) results.

```
train.py / eval.py                  entry points
diffusion_policy/
  config/                           hydra configs: 2 workspaces + 20 tasks
    task/<task>_{lowdim,image}[_abs].yaml
  workspace/                        training loops
  policy/                           diffusion policies
  model/                            UNet, EMA, normalizer, crop randomizer
  dataset/                          robomimic hdf5 -> replay buffer -> sequence sampler
  env/robomimic/, env_runner/,
  gym_util/                         rollout / evaluation in robosuite
  scripts/robomimic_dataset_conversion.py   delta -> absolute action conversion
```

## Data

Download the robomimic datasets as packaged by Diffusion Policy into `data/` (the configs expect `data/robomimic/datasets/<task>/ph/{low_dim,image}[_abs].hdf5`):

```console
$ mkdir -p data && cd data
$ wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip   # ~1.9 GB
$ wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip    # ~85 GB
$ unzip robomimic_lowdim.zip && unzip robomimic_image.zip && cd ..
```

## Training

Task configs come in four flavours per task: `<task>_lowdim`, `<task>_lowdim_abs`, `<task>_image`, `<task>_image_abs`. The `_abs` variants (absolute end-effector actions, rotation as 6D) are the ones used for the paper results.

```console
# state
$ python train.py --config-name=train_diffusion_unet_lowdim_workspace task=square_lowdim_abs \
    training.seed=42 training.device=cuda:0

# image
$ python train.py --config-name=train_diffusion_unet_hybrid_workspace task=square_image_abs \
    training.seed=42 training.device=cuda:0
```

Logging goes to [wandb](https://wandb.ai) (`wandb login` first, or add `logging.mode=offline`). Outputs (config, logs, checkpoints, rollout videos) are written to `data/outputs/<date>/<time>_<name>_<task>/`.

Quick end-to-end smoke test (a few steps of training, validation, a short rollout and checkpointing):

```console
$ python train.py --config-name=train_diffusion_unet_hybrid_workspace task=square_image_abs \
    training.debug=True logging.mode=disabled \
    task.env_runner.n_train=1 task.env_runner.n_test=2 task.env_runner.n_envs=null task.env_runner.max_steps=20
```

Evaluation runs 50 test rollouts on 28 parallel environments by default (about 1 GB of RAM per image environment).

## Evaluation of a checkpoint

```console
$ python eval.py -c data/outputs/.../checkpoints/latest.ckpt -o data/eval_output -d cuda:0
```

## License and acknowledgement

Released under the MIT license; see [LICENSE](LICENSE). Original code © 2023 Columbia Artificial Intelligence and Robotics Lab. `diffusion_policy/codecs/imagecodecs_numcodecs.py` is © Christoph Gohlke under the BSD-3-Clause license (see the file header).

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
