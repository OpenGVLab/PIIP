#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
QUOTATYPE=${QUOTATYPE:-"reserved"}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTATYPE} \
    ${SRUN_ARGS} \
    python -u main.py \
    --model ${CONFIG} \
    --batch-size 128 \
    --lr 3e-5 \
    --epochs 20 \
    --weight-decay 0.1 \
    --reprob 0.0 \
    --seed 0 \
    --unscale-lr \
    --no-repeated-aug \
    --from_scratch_lr_ratio 10 \
    --data-path /mnt/lustre/share/images \
    --output_dir exp ${@:4}