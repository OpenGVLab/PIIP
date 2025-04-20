# PIIP-LLaVA

This folder contains code for PIIP-LLaVA, developed on top of [LLaVA-1.5](https://github.com/haotian-liu/LLaVA).

The released model weights are provided in [**the parent folder**](../README.md) and [**Huggingface**](https://huggingface.co/collections/OpenGVLab/piip-6804939a32e695f42cf3f227).


# Installation

1. Clone this repo:
  ```bash
  git clone https://github.com/OpenGVLab/PIIP
  cd PIIP/llava/
  ```

2. Create a conda virtual environment and activate it:

  ```bash
  conda create -n piip_llava python=3.10 -y
  conda activate piip_llava
  ```

3. Install packages
  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  pip install -e ".[train]"
  pip install flash-attn==2.3.6 --no-build-isolation
  # install deformable attention
  cd llava/model/multimodal_encoder/piip/ops && sh compile.sh
  ```

4. For pretrained models, the following huggingface and timm models will be downloaded automatically (you can also download them manually): 

    lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5, OpenGVLab/clip-vit-large-patch14to16-224, OpenGVLab/clip-vit-large-patch14to16-336, openai/clip-vit-base-patch16, convnext_base.clip_laiona_augreg_320, convnext_large_mlp.clip_laion2b_ft_320

5. Prepare the training and evaluation datasets according to the [LLaVA-1.5 guidelines] (https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train). The `playground` folder should be like this:

    ```
    .
    └── data
        ├── coco
        ├── eval
        │   ├── gqa
        │   │   ├── data
        │   │   └── llava_gqa_testdev_balanced.jsonl
        │   ├── mmbench
        │   │   └── mmbench_dev_20230712.tsv
        │   ├── mm-vet
        │   │   ├── images
        │   │   └── llava-mm-vet.jsonl
        │   ├── pope
        │   │   ├── coco
        │   │   └── llava_pope_test.jsonl
        │   ├── scienceqa
        │   │   ├── images
        │   │   ├── llava_test_CQM-A.json
        │   │   ├── pid_splits.json
        │   │   └── problems.json
        │   ├── seed_bench
        │   │   ├── extract_video_frames.py
        │   │   ├── llava-seed-bench.jsonl
        │   │   ├── preprocess_video_frames.py
        │   │   ├── SEED-Bench-image
        │   │   ├── SEED-Bench.json
        │   │   ├── SEED-Bench-video-image
        │   │   └── SEED-Bench-video-image-source
        │   ├── textvqa
        │   │   ├── llava_textvqa_val_v051_ocr.jsonl
        │   │   └── TextVQA_0.5.1_val.json
        │   └── vqav2
        │       ├── llava_vqav2_mscoco_test2015.jsonl
        │       ├── llava_vqav2_mscoco_test-dev2015.jsonl
        │       └── test2015
        ├── gqa
        ├── LLaVA-Pretrain
        ├── llava_v1_5_mix665k.json
        ├── ocr_vqa
        ├── textvqa
        └── vg
    ```
    

Note: the core model code is under `llava/model/multimodal_encoder`.

## Training

To train PIIP models, change the variables in `piip_pretrain.sh` and `piip_finetune.sh` and run:

```bash
bash shell_scripts/piip_pretrain.sh

bash shell_scripts/piip_finetune.sh
```

To train the LLaVA-1.5 baseline, change the variables in `llava1_5_pretrain.sh` and `llava1_5_finetune.sh` and run:

```bash
bash shell_scripts/llava1_5_pretrain.sh

bash shell_scripts/llava1_5_finetune.sh
```

Training runs on 8 A100 (80G) GPUs. If OOM is encountered, try Zero3 or larger `gradient_accumulation_steps` while keeping the product of `gradient_accumulation_steps` and `per_device_train_batch_size` unchanged.

To use slurm for training, change `torchrun --nproc_per_node=8 --master_port=12345 llava/train/train_mem.py` in the scripts to `srun -p xxx --job-name=xxx --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=12 --ntasks=1 --kill-on-bad-exit=1 deepspeed llava/train/train_mem.py`.

> [!Note]
> `transformers<4.49.0` automatically changes the keys with name `gamma` to `weight` when loading pretrained models, but ConvNeXt in timm uses `gamma` as a parameter. 
> To fix this legacy issue, we use a [monkey patch](llava/model/timm_convnext_monkey_patch.py) to change the parameter name in timm. 
> This may also be solved by using `transformers>=4.49.0`, but our pretrained models and the original LLaVA-1.5 is based on `transformers==4.37.2`, and newer version could potentially leads to other issues.

## Evaluation

To evaluate PIIP or LLaVA-1.5 models on all benchmarks, change the `CHECKPOINT_PATH` in `eval.sh` and run:

```bash
bash shell_scripts/eval.sh
```

For MMBench, submit the result file in `eval_results/mmbench/` to [the evaluation server](https://opencompass.org.cn/leaderboard-multimodal).

For MMVet, submit the result file in `eval_results/mm-vet/` to [the evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) or use the official jupyter notebook.

For VQAv2, submit the result file in `eval_results/vqav2/` to [the evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).


Evaluation runs on 1 A100 (80G) GPU. For more details, refer to [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## Inference Demo

First download the pretrained checkpoints from [here](https://github.com/OpenGVLab/PIIP#multimodal-understanding).

To use gradio for inference (recommended, faster as model is loaded only once):


```bash
python gradio_demo.py --model_path PATH/TO/CHECKPOINT_FILE
```

To use command line for inference:

```bash
python inference.py --model_path PATH/TO/CHECKPOINT_FILE --img_path images/llava_logo.png --prompt "Describe the image."
```



## FLOPs Calculation

We provide a simple script to calculate the number of FLOPs. Change the `config_list` in `get_flops_llava.py` and run:

```bash
python get_flops_llava.py
```

Then the FLOPs and number of parameters are recorded in `flops_llava.txt`.


## Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{piip,
  title={Parameter-Inverted Image Pyramid Networks},
  author={Zhu, Xizhou and Yang, Xue and Wang, Zhaokai and Li, Hao and Dou, Wenhan and Ge, Junqi and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2406.04330},
  year={2024}
}

@article{piip_v2,
  title={Parameter-Inverted Image Pyramid Networks for Visual Perception and Multimodal Understanding},
  author={Wang, Zhaokai and Zhu, Xizhou and Yang, Xue and Luo, Gen and Li, Hao and Tian, Changyao and Dou, Wenhan and Ge, Junqi and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2501.07783},
  year={2025}
}
```
