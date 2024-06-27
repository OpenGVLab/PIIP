# PIIP for Semantic Segmentation

This folder contains code for applying PIIP on semantic segmentation, developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

The released model weights are provided in [**the parent folder**](../README.md).

## Installation

Please refer to [installation of object detection](../mmdetection/README.md). 

Then link the `ops` and `pretrained` directories to this folder:

```bash
ln -s ../mmdetection/ops .
ln -s ../mmdetection/pretrained .
```

Note: the core model code is under `mmseg/models/backbones/`.

## Training

To train PIIP-H6B UperNet on ADE20K on a single node with 8 gpus for 12 epochs run:

```bash
sh tools/dist_train.sh configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py 8
# or manage jobs with slurm
GPUS=8 sh tools/slurm_train.sh <partition> <job-name> configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py
```

## Evaluation

To evaluate PIIP-H6B UperNet on ADE20K on a single node with a single gpu:

```bash
# w/ deepspeed
python tools/test.py configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/iter_80000/global_step80000 --eval mIoU
# w/ deepspeed and set `deepspeed=False` in the configuration file
python tools/test.py configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/iter_80000/global_step80000/mp_rank_00_model_states.pt --eval mIoU
# w/o deepspeed
python tools/test.py configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.pth --eval mIoU
```

```bash
# w/ deepspeed
sh tools/dist_test.sh configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/iter_80000/global_step80000 8 --eval mIoU
# w/ deepspeed and set `deepspeed=False` in the configuration file
sh tools/dist_test.sh configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/iter_80000/global_step80000/mp_rank_00_model_states.pt 8 --eval mIoU
# w/o deepspeed
sh tools/dist_test.sh configs/piip/2branch/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.py work_dirs/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5/upernet_internvit_h6b_256_512_80k_ade20k_bs16_lr4e-5.pth 8 --eval mIoU
```

## FLOPs calculation

We provide a simple script to calculate the number of FLOPs. Change the `config_list` in `../classification/get_flops.py` and run

```bash
# use the classification environment
cd ../classification/
python get_flops.py
```

Then the FLOPs and number of parameters are recorded in `flops.txt`.

## Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{piip,
  title={Parameter-Inverted Image Pyramid Networks},
  author={Zhu, Xizhou and Yang, Xue and Wang, Zhaokai and Li, Hao and Dou, Wenhan and Ge, Junqi and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2406.04330},
  year={2024}
}
```
