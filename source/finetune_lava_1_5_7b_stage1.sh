#!/bin/bash


export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LIBRARY_PATHH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"


WANDB_API_KEY=bcd2d5eb473ab8f9d49469b13d8c457c80087b98

img_path='/sdb/public_data/dataset/uni_food/images/train'
data_path='/sdb/public_data/dataset/uni_food/final/train.json'


# img_path=''
# data_path=''


experiment_name='your_exp_name'


bs=4
gradient_accumulation_steps=8


tune_mm_mlp_adapter=True
vision_projector_lr=0.001
load_pretrain_VCE=False

lora_enable='False'
lora_ranks=32
lora_alpha=64
lr=0


output_dir='your_path_to/'$experiment_name

mkdir -p $output_dir


nohup  deepspeed --master_port 29526 --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --lora_enable $lora_enable --lora_ranks $lora_ranks --lora_alpha $lora_alpha --mm_projector_lr $vision_projector_lr \
    --tune_mm_mlp_adapter $tune_mm_mlp_adapter \
    --load_pretrain_VCE $load_pretrain_VCE \
    --model_name_or_path /sdb/public_data/model_zoo/lava_modify/llava-v1.5-7b \
    --deepspeed ./scripts/zero2.json \
    --version v1 \
    --data_path $data_path \
    --image_folder $img_path \
    --vision_tower /sdb/public_data/model_zoo/openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --VCE_pretrain_path $VCE_pretrain_path \
    --image_aspect_ra pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir $output_dir \
    --num_train_epochs 5 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 9999999 \
    --save_total_limit 5 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    > train_log/$experiment_name.log 2>&1 &


