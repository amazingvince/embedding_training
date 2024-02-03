#!/bin/bash
set -e

# configure environment variables
export TOKENIZERS_PARALLELISM=true
export WANDB_ENTITY="amazingvince"
export WANDB_WATCH="gradients"
export WANDB_PROJECT="long-encoder"
echo "wandb watching: $WANDB_WATCH"

# Get the number of processors using nproc and subtract 2
NUM_PROCS=$(nproc)
NUM_WORKERS=$((NUM_PROCS - 20))
# Ensure NUM_WORKERS is not negative
if [ $NUM_WORKERS -lt 1 ]; then
    NUM_WORKERS=1
fi
echo "Number of workers: $NUM_WORKERS"

# Script arguments
HF_MODEL_TAG="amazingvince/bert_DupMAE"
TOKENIZER_NAME="bert-base-uncased"
CONFIG_NAME="config.json"
DATASET_TAG="JeanKaddour/minipile"
DATASET_CFG="default"
MAX_SOURCE_LEN=1024


SHORT_NAME="$(basename $HF_MODEL_TAG)"
DS_SHORT_NAME="$(basename $DATASET_TAG)"
MASK_RATIO=0.25
RUN_NAME="$SHORT_NAME-${DS_SHORT_NAME}_${MAX_SOURCE_LEN}-vN"
RUNTIME_DIR="./runtime/masked/outputs-$RUN_NAME"
LOGGING_DIR="$RUNTIME_DIR/logs"
RUN_SEED=$RANDOM

NUM_EPOCHS=1
BATCH_SIZE=16
EVAL_BATCH_SIZE=16
WARMUP_STEPS=100
WEIGHT_DECAY=0.01

# Optimizer
OPTIMIZER_ID="adamw_torch_fused" #paged_adamw_32bit" # "paged_adamw_32bit"
#OPTIMIZER_ID="paged_adamw_32bit" # "paged_adamw_32bit"
BETA1=0.90
BETA2=0.99
ADAM_EPS=1e-8
LR_SCHEDULER_TYPE="inverse_sqrt"
LEARNING_RATE=2.5e-4
MAX_GRAD_NORM=1.0
GC_STEPS=16
GRAD_CHKPTING=False

# Checkpointing and logging
EVAL_STRATEGY="steps"
EVAL_STEPS=150
MAX_EVAL_SAMPLES=300

LOGGING_STEPS=5
SAVE_STRATEGY="steps"
CHK_STEPS=100
SAVE_LIMIT=1
REPORT_TO="wandb"

# Data type
DATA_TYPE="--bf16 --bf16_full_eval False"
USE_TF32=True

DEEPSPEED_CONFIG="config/deepspeed/ds_config_zero2_bf16.json"

mkdir -p $RUNTIME_DIR $LOGGING_DIR

# --config_name 
#     --model_name_or_path $HF_MODEL_TAG \
#   --gradient_accumulation_steps $GC_STEPS \

# Train
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 1 ./run_mlm.py \
    --config_name $CONFIG_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --dataset_name "$DATASET_TAG" \
    --dataset_config_name "$DATASET_CFG" \
    --do_train \
    --do_eval \
    --save_strategy $SAVE_STRATEGY \
    --evaluation_strategy $EVAL_STRATEGY --eval_steps $EVAL_STEPS \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --dataloader_drop_last False \
    --dataloader_num_workers $NUM_WORKERS \
    --dataloader_pin_memory True \
    --gradient_checkpointing $GRAD_CHKPTING \
    --hub_model_id $RUN_NAME --hub_strategy "every_save" \
    --hub_private_repo True \
    --learning_rate $LEARNING_RATE \
    --load_best_model_at_end False \
    --logging_dir $RUNTIME_DIR \
    --logging_steps $LOGGING_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_seq_length $MAX_SOURCE_LEN \
    --mlm_probability $MASK_RATIO \
    --num_train_epochs $NUM_EPOCHS \
    --optim $OPTIMIZER_ID \
    --adam_epsilon $ADAM_EPS \
    --adam_beta2 $BETA2 \
    --output_dir $RUNTIME_DIR \
    --overwrite_output_dir \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --prediction_loss_only False \
    --preprocessing_num_workers $NUM_WORKERS \
    --push_to_hub --save_safetensors \
    --report_to $REPORT_TO \
    --run_name $RUN_NAME \
    --save_steps $CHK_STEPS \
    --save_total_limit $SAVE_LIMIT \
    --seed $RUN_SEED \
    --tf32 $USE_TF32 \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    $DATA_TYPE \
    # --torch_compile_backend inductor \


# --whole_word_mask \
#    --pad_to_max_length True \

wandb artifact put -n training-scripts -t code -a $RUN_NAME $0
