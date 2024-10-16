#!/bin/bash
# always change the log file name
logfile=./logs/camera_motion/train.log
# rename the old output dir if it's a run with the same parameters unless to resume the training

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=1

nohup torchrun --nproc_per_node=auto \
    --master_port=34651 \
    train.py \
    --cfg-path lavis/projects/malmm/cls_camera_motion.yaml \
    --options \
    model.vit_precision fp16 \
    model.num_frames 100 \
    run.init_lr 1e-4 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.num_workers 2 \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    run.max_epoch 100 \
    run.num_beams 5 \
    run.accum_grad_iters 1 \
    run.seed 42 \
    run.evaluate True \
    run.report_metric True \
    run.prefix train \
    > $logfile 2>&1 &
    # run.resume_ckpt_path \

cursor $logfile
