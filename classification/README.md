# PIIP for Classification

This folder contains code for applying PIIP on image classification.

The released model weights are provided in [**the parent folder**](../README.md).

## Installation

**Note:** environment for classification is different from the detection and segmentation environment.

- Clone this repo:

  ```bash
  git clone https://github.com/OpenGVLab/PIIP
  cd PIIP/
  ```
- Create a conda virtual environment and activate it ():

  ```bash
  conda create -n piip_cls python=3.9 -y
  conda activate piip_cls
  ```
- Install dependencies

  ```bash
  pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install timm==0.5.4
  pip install mmcv-full==1.4.2
  pip install flash_attn==0.2.8
  pip install einops
  # install deformable attention
  cd ops && sh compile.sh
  ```
- (Optional) to use FusedMLP, install flash attention from source, and set `mlp_type="fused_mlp"` of each branch in the config file
  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v0.2.8
  pip install ninja
  python setup.py install
  cd csrc/fused_dense_lib
  pip install .
  ```
- Download pretrained ViT weights from [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md) and [augreg](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py):

  ```bash
  mkdir pretrained
  cd pretrained
  # DeiT checkpoints
  wget https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
  wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
  wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
  # Augreg checkpoints
  wget https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz
  wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
  wget https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz
  # convert Augreg checkpoints into .pth
  cd ..
  sh convert_augreg_models.sh
  ```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Training

To train a PIIP-SBL on ImageNet on a single node with 8 gpus for 20 epochs run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env -u main.py --model piip_3branch_sbl_384-192-128_cls_token_augreg.py --data-path /path/to/imagenet --output_dir exp --batch-size 128 --lr 3e-5 --epochs 20 --weight-decay 0.1 --reprob 0.0 --seed 0 --unscale-lr --no-repeated-aug --from_scratch_lr_ratio 10
# or manage jobs with slurm
GPUS=8 sh slurm_train.sh <partition> <job-name> piip_3branch_sbl_384-192-128_cls_token_augreg.py
```

## Evaluation

To evaluate a pretrained PIIP-SBL on ImageNet val with a single GPU run:

```bash
python -u main.py --eval --resume /path/to/piip_3branch_sbl_384-192-128_cls_token_augreg.pth --model piip_3branch_sbl_384-192-128_cls_token_augreg.py --data-path /path/to/imagenet
```

This should give

```
* Acc@1 85.862 Acc@5 97.870 loss 0.615
```

## FLOPs Calculation

We provide a simple script to calculate the number of FLOPs. Change the `config_list` in `get_flops.py` and run

```bash
python get_flops.py
```

Then the FLOPs and number of parameters are recorded in `flops.txt`.


## Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{piip,
  title={Parameter-Inverted Image Pyramid Networks},
  author={Zhu, Xizhou and Yang, Xue and Wang, Zhaokai and Li, Hao and Dou, Wenhan and Ge, Junqi and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```
