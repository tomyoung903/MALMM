#!/bin/bash
# always change the log file name

timestamp=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
run_id=${timestamp}_scoring_in_camera_motion_style
logfile=./logs/camera_motion/train_${run_id}.log
# rename the old output dir if it's a run with the same parameters unless to resume the training

random_port=$((49152 + RANDOM % 16384))

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

nohup torchrun --nproc_per_node=auto \
    --master_port=${random_port} \
    train.py \
    --cfg-path lavis/projects/malmm/cls_scoring_in_camera_motion_style.yaml \
    --options \
    model.vit_precision fp32 \
    model.num_frames 20 \
    run.init_lr 1e-7 \
    run.batch_size_train 12 \
    run.batch_size_eval 24 \
    run.num_workers 2 \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.freeze_vit False \
    model.memory_bank_length 20 \
    run.max_epoch 100 \
    run.num_beams 5 \
    run.accum_grad_iters 1 \
    run.seed 42 \
    run.evaluate False \
    run.report_metric True \
    run.prefix ${run_id} \
    > $logfile 2>&1 &
    # run.resume_ckpt_path \

cursor $logfile

# /home/tom/Open-Sora-dev/tools/scoring/MA-LMM/train_camera_motion.sh
