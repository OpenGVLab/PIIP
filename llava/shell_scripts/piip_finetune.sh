set -x

#! Directory of pretrained checkpoint
PRETRAIN_RUN_NAME=checkpoints/7f_piip-llava_clip-bl_512-448_7B_pretrain_1node_zero2

#! LLM size
LLM="7B"
# LLM="13B"

#! Deepspeed Zero
# ZERO=zero2
ZERO=zero3

#! Gradient accumulation
GRAD_ACC=4
# GRAD_ACC=1
# GRAD_ACC=2


BATCH_SIZE=$((16 / GRAD_ACC))

# find vision tower config file
files=($(ls ${PRETRAIN_RUN_NAME}/*.py 2>/dev/null))
if [ ${#files[@]} -eq 0 ]; then
  echo "Error: No .py files found in the directory '${PRETRAIN_RUN_NAME}'."
  exit 1
elif [ ${#files[@]} -gt 1 ]; then
  echo "Error: Multiple .py files found in the directory '${PRETRAIN_RUN_NAME}':"
  for file in "${files[@]}"; do
    echo "$file"
  done
  exit 1
else
  echo "Single .py file found: ${files[0]}"
fi

VISION_TOWER=${files[0]}
VISION_TOWER=configs/$(basename $VISION_TOWER)


if [ "$LLM" = "7B" ]; then
    LLM_PATH="lmsys/vicuna-7b-v1.5"
elif [ "$LLM" = "13B" ]; then
    LLM_PATH="lmsys/vicuna-13b-v1.5"
else
    echo "Invalid LLM value. Please use '7B' or '13B'."
    exit 1
fi


if [ "$ZERO" = "zero2" ]; then
    ZERO_CONFIG="./scripts/zero2_overlap_comm_false.json"
elif [ "$ZERO" = "zero3" ]; then
    ZERO_CONFIG="./scripts/zero3.json"
else
    echo "Invalid ZERO value."
    exit 1
fi


RUN_NAME="${PRETRAIN_RUN_NAME#checkpoints/}_${ZERO}_grad-acc${GRAD_ACC}_ft"

OUTPUT_DIR=./checkpoints/$RUN_NAME
mkdir -p $OUTPUT_DIR
cp $VISION_TOWER $OUTPUT_DIR/


#! Without Wandb
export WANDB_MODE=disabled
#! With Wandb: uncomment the following lines
# export WANDB_API_KEY="your api key"
# export WANDB_ENTITY='your wandb entity'
# export WANDB_PROJECT='your wandb project'
# export WANDB_RUN_NAME=$RUN_NAME
# export WANDB_LOG_MODEL=false


timestamp=$(date +"%Y%m%d_%H%M%S")


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export PYTHONPATH=.


torchrun --nproc_per_node=8 --master_port=12345 llava/train/train_mem.py \
    --deepspeed $ZERO_CONFIG \
    --model_name_or_path $LLM_PATH \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower $VISION_TOWER \
    --pretrain_mm_mlp_adapter $PRETRAIN_RUN_NAME/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    2>&1 | tee $OUTPUT_DIR/finetune_${timestamp}.log
