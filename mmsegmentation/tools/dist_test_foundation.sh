#!/usr/bin/env bash

/usr/local/lib/miniconda3/condabin/conda init bash
source activate
conda activate /mnt/afs/user/chenzhe/.conda/envs/vitdet

cd /mnt/afs/user/chenzhe/workspace/ViTDetection/mmsegmentation

pip install tensorboard

export PYTHONPATH=$PYTHONPATH:/mnt/afs/user/chenzhe/workspace/ViTDetection/mmsegmentation
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
# use 8 gpus
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} --master_port=63667 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
