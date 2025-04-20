set -x

#! Path of vision tower (pretrained model)
VISION_TOWER=openai/clip-vit-large-patch14-336

#! LLM size
LLM="7B"
# LLM="13B"

#! Deepspeed Zero
ZERO=zero2
# ZERO=zero3


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


config_name=$(basename "$VISION_TOWER")
RUN_NAME="${config_name}_${LLM}_pretrain_1node_${ZERO}"

OUTPUT_DIR=./checkpoints/$RUN_NAME
mkdir -p $OUTPUT_DIR


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
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    2>&1 | tee $OUTPUT_DIR/pretrain_${timestamp}.log
