#!/bin/bash
# always change the log file name

timestamp=$(date '+%Y-%m-%d_%Hh%Mm%Ss')
logfile=./logs/camera_motion/train_${timestamp}_debug.log
# rename the old output dir if it's a run with the same parameters unless to resume the training

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1

nohup torchrun --nproc_per_node=auto \
    --master_port=34653 \
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
    run.prefix ${timestamp}_debug \
    > $logfile 2>&1 &
    # run.resume_ckpt_path \

cursor $logfile

# /home/tom/Open-Sora-dev/tools/scoring/MA-LMM/train_camera_motion.sh
