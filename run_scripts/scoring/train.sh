#!/bin/bash
# always change the log file name
logfile=./logs/scoring/train_lr_1e-7_unfreeze_vit_vit_fp32_3_classes.log
# rename the old output dir if it's a run with the same parameters
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export dbad=1025
# bash ./run_scripts/scoring/train.sh > $logfile 2>&1 &

torchrun --nproc_per_node=auto \
    --master_port=34651 \
    train.py \
    --cfg-path lavis/projects/malmm/cls_scoring.yaml \
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
    run.prefix train \
    > $logfile 2>&1 &
    # run.resume_ckpt_path \

sleep 10
code $logfile
