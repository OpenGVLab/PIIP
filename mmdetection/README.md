# PIIP for Detection

This folder contains code for applying PIIP on object detection and instance segmentation, developed on top of [MMDetection v2.25.3](https://github.com/open-mmlab/mmdetection/tree/v2.25.3).

The released model weights are provided in [**the parent folder**](../README.md).

## Installation

- Clone this repo:

  ```bash
  git clone https://github.com/OpenGVLab/PIIP
  cd PIIP/
  ```
- Create a conda virtual environment and activate it:

  ```bash
  conda create -n piip python=3.9 -y
  conda activate piip
  ```
- Install `PyTorch>=1.11<2.0` and `torchvision>=0.13.0` with `CUDA>=10.2`:

  For example, to install torch==1.12.0 with CUDA==11.3:

  ```bash
  pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```
- Install `flash-attn==0.2.8` :

  If you want to fully replicate my results, please install `v0.2.8`, otherwise install the latest version.

  This is because different versions of flash attention yield slight differences in results.

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v0.2.8
  pip install ninja
  python setup.py install # I use gcc-7.3 to compile this package
  # for FusedMLP
  cd csrc/fused_dense_lib
  pip install .
  ```
- Install other requirements:

  ```bash
  conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
  pip install opencv-python
  pip install timm==0.6.11
  pip install yapf==0.40.1
  pip install addict
  pip install deepspeed==0.8.0 # please install this old version
  pip install pydantic==1.10.2 # later versions may have compatibility issues
  ```
- Install our customized `mmcv-full==1.7.0`:

  ```bash
  cd mmcv/
  export MMCV_WITH_OPS=1
  python setup.py develop
  cd ../
  ```
- Install our customized mmdetection:

  ```bash
  cd mmdetection/
  python setup.py develop
  ```
- Compile the deformable attention:

  ```bash
  cd mmdetection/ops
  sh compile.sh
  ```
- Selectively download pretrained ViT weights from [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), [DeiT III](https://github.com/facebookresearch/deit/blob/main/README_revenge.md), [MAE](https://github.com/facebookresearch/mae), [InternVL](https://github.com/OpenGVLab/InternVL-MMDetSeg/blob/main/mmsegmentation/README.md), [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), [DINOv2](https://github.com/facebookresearch/dinov2), [AugReg](https://github.com/google-research/vision_transformer) and [Uni-Perceiver](https://github.com/fundamentalvision/Uni-Perceiver):

  ```bash
  mkdir mmdetection/pretrained
  cd mmdetection/pretrained
  # Default: DeiT III and DeiT
  wget https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
  wget https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth
  wget https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth
  wget https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth
  # Optional - for InternViT-6B experiments
  wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth
  wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px.pth
  # Optional - for different pretraining methods
  # - MAE
  wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
  wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
  # - BEiTv2
  wget https://github.com/addf400/files/releases/download/BEiT-v2/beitv2_base_patch16_224_pt1k_ft21k.pth
  wget https://github.com/addf400/files/releases/download/BEiT-v2/beitv2_large_patch16_224_pt1k_ft21k.pth
  # - DINOv2
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
  # - AugReg
  wget https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.pth?download=true
  wget https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.pth?download=true
  wget https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth?download=true
  wget https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth?download=true
  # - Uni-Perceiver
  wget https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/uni-perceiver-base-L12-H768-224size-torch-pretrained_converted.pth
  wget https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth
  ```
- Prepare the COCO dataset according to the [MMDetection guidelines](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#prepare-datasets).

Note: the core model code is under `mmdet/models/backbones/`.

## Training

To train PIIP-TSB Mask R-CNN on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```bash
sh tools/dist_train.sh configs/piip/3branch/mask_rcnn_beit_tsb_1120_896_448_fpn_1x_coco_bs16.py 8
# or manage jobs with slurm
GPUS=8 sh tools/slurm_train.sh <partition> <job-name> configs/piip/3branch/mask_rcnn_beit_tsb_1120_896_448_fpn_1x_coco_bs16.py
```

**Note**: You can modify the `deepspeed` parameter in the configuration file to decide whether to use deepspeed. If you want to resume the deepspeed pretrained model for finetuning, you need to set `deepspeed_load_module_only=True` in the config.

## Evaluation

To evaluate PIIP-H6B Mask R-CNN on COCO val2017 on a single node with a single gpu:

```bash
# w/ deepspeed
python tools/test.py configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/iter_87961/global_step87960 --eval bbox segm
# w/ deepspeed and set `deepspeed=False` in the configuration file
python tools/test.py configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/iter_87961/global_step87960/mp_rank_00_model_states.pt --eval bbox segm
# w/o deepspeed
python tools/test.py configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth --eval bbox segm
```

To evaluate PIIP-H6B Mask R-CNN on COCO val2017 on a single node with 8 gpus:

```bash
# w/ deepspeed
sh tools/dist_test.sh configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/iter_87961/global_step87960 8 --eval bbox segm
# w/ deepspeed and set `deepspeed=False` in the configuration file
sh tools/dist_test.sh configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/iter_87961/global_step87960/mp_rank_00_model_states.pt 8 --eval bbox segm
# w/o deepspeed
sh tools/dist_test.sh configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms work_dirs/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth 8 --eval bbox segm

```

This should give

```bash
Evaluating bbox...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.773
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.614
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.396
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.605
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.715
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.808

Evaluating segm...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.743
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.754
```

## FLOPs calculation

We provide a simple script to calculate the number of FLOPs. Change the `config_list` in `../classification/get_flops.py` and run

```bash
# use the classification environment
conda activate piip_cls
cd ../classification/
python get_flops.py
```

Then the FLOPs and number of parameters are recorded in `flops.txt`.

## Faster Training

- Use `FusedMLP` and discard `Projection` and `LayerNorm` in interactions by setting `with_proj=False`, `norm_layer="none"` of backbone and `mlp_type="fused_mlp"` of each branch in the config file. These changes do not affect performance.


|   Backbone   |  Detector  |        Pretrain        |  Resolution  | Schd |  Box mAP  | Mask mAP | #Training Time | #FLOPs | #Param |                                                                                                                                                                                               Download                                                                                                                                                                                               |
| :----------: | :--------: | :---------------------: | :-----------: | :--: | :-------: | :-------: | :------------: | :----: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PIIP-SBL w/o | Mask R-CNN |         AugReg         | 1568/1120/672 |  1x  |   48.3   |   42.6   |      23h      | 1874G |  498M  |             [log](https://huggingface.co/OpenGVLab/PIIP/raw/main/detection/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16.log.json) \| [ckpt](https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16.pth?download=true) \| [cfg](mmdetection/configs/piip/3branch/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16.py)             |
| PIIP-SBL w/ | Mask R-CNN |         AugReg         | 1568/1120/672 |  1x  | `running` | `running` |      21h      | 1741G |  470M  | [log](https://huggingface.co/OpenGVLab/PIIP/raw/main/detection/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16_speedup.log.json) \| [ckpt](https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16_speedup.pth?download=true) \| [cfg](mmdetection/configs/piip/3branch/mask_rcnn_augreg_sbl_1568_1120_672_fpn_1x_coco_bs16_speedup.py) |
| InternViT-6B | Mask R-CNN |        InternVL        |     1024     |  1x  |   53.8   |   48.1   |     3d19h     | 29323G | 5919M |                          [log](https://huggingface.co/OpenGVLab/PIIP/raw/main/detection/mask_rcnn_internvit_6b_fpn_1x_coco_bs16_ms.log.json) \| [ckpt](https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/mask_rcnn_internvit_6b_fpn_1x_coco_bs16_ms.pth?download=true) \| [cfg](mmdetection/configs/piip/baseline/mask_rcnn_internvit_6b_fpn_1x_coco_bs16_ms.py)                          |
| PIIP-H6B w/o | Mask R-CNN | MAE (H) + InternVL (6B) |   1024/512   |  1x  |   55.8   |   49.0   |     3d14h     | 11080G | 6872M |           [log](https://huggingface.co/OpenGVLab/PIIP/raw/main/detection/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.log.json) \| [ckpt](https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth?download=true) \| [cfg](mmdetection/configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.py)           |
| PIIP-H6B w/ | Mask R-CNN | MAE (H) + InternVL (6B) |   1024/512   |  1x  | `running` | `running` |     2d14h     | 10692G | 6752M |           [log](https://huggingface.co/OpenGVLab/PIIP/raw/main/detection/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.log.json) \| [ckpt](https://huggingface.co/OpenGVLab/PIIP/resolve/main/detection/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.pth?download=true) \| [cfg](mmdetection/configs/piip/2branch/mask_rcnn_internvit_h6b_1024_512_fpn_1x_coco_bs16_ms.py)           |

Note that the training speed of multi-branch models can be affected by many factors, e.g. GPU utilization or optimization techniques. Further optimization might be necessary to enhance training efficiency.

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
