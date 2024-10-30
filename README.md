# MSI-NeRF
Linking Omni-Depth with View Synthesis through Multi-Sphere Image aided Generalizable Neural Radiance Field

### [Paper](https://arxiv.org/abs/2403.10840) | [Video](https://www.youtube.com/watch?v=UmtEM4Uj5b0)

> MSI-NeRF <br />
> Dongyu Yan, Guanyu Huang, Fengyu Quan and Haoyao Chen <br />
> WACV 2025

<p align="center">
  <img width="100%" src="./assets/main.png"/>
</p>

This repository contains code for the paper MSI-NeRF: Linking Omni-Depth with View Synthesis through Multi-Sphere Image aided Generalizable Neural Radiance Field..

## Install

```sh
conda env create -f environment.yml
conda activate msinerf
```

## Data and Checkpoint

The Replica360 dataset proposed in the paper can be downloaded [here](https://drive.google.com/file/d/19sy-B9-rViCCnUaXXpRrh0sY7SBkQQ9r/view?usp=sharing) (Only a sample case for testing, full version coming soon!)

The checkpoint can be downloaded [here](https://drive.google.com/file/d/15nYvTNBsvIXQ1jq54mQ5XgEzUJ_5WNx3/view?usp=sharing)

## Usage

### Train

```sh
python train.py --config ./config/default.txt --gpus 4 --batch_size 1 --split_ratio 0.95 --max_epoch 30
```

### Test

1. Run depth map generation (along with color map generation in rig center)

```sh
python test.py --config ./config/default.txt --gpus 1 --batch_size 1 --ckpts_epoch 29 --split_ratio 0.95
```

1. Run color map generation in NVS dataset

```sh
python test.py --config ./config/default.txt --gpus 1 --batch_size 1 --ckpts_epoch 29 --eval_nvs
```

3. Run novel view synthesis using predefined trajectory

```sh
python test.py --config ./config/default.txt --gpus 1 --batch_size 1 --ckpts_epoch 29 --traj_type 0 --render_novel_view --eval_nvs (optional)
```

if you find this paper useful, please cite:

```bibtex
@article{yan2024msi,
  title={MSI-NeRF: Linking Omni-Depth with View Synthesis through Multi-Sphere Image aided Generalizable Neural Radiance Field},
  author={Yan, Dongyu and Huang, Guanyu and Quan, Fengyu and Chen, Haoyao},
  journal={arXiv preprint arXiv:2403.10840},
  year={2024}
}
```
