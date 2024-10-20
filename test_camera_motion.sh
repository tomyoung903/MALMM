#!/bin/bash
# always change the log file name

logfile=./logs/camera_motion/test_2024-10-20_15h40m07s.log
# rename the old output dir if it's a run with the same parameters unless to resume the training
timestamp='2024-10-20_15h40m07s'
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2

nohup torchrun --nproc_per_node=auto \
    --master_port=34651 \
    train.py \
    --cfg-path lavis/projects/malmm/camera_motion_test.yaml \
    --options \
    model.num_frames 20 \
    run.batch_size_eval 24 \
    run.num_workers 2 \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned True \
    model.load_pretrained False \
    model.num_query_token 32 \
    model.memory_bank_length 20 \
    run.num_beams 5 \
    run.seed 42 \
    run.evaluate True \
    run.report_metric True \
    run.prefix $timestamp \
    > $logfile 2>&1 &
    
    # run.resume_ckpt_path \

cursor $logfile

# /home/tom/Open-Sora-dev/tools/scoring/MA-LMM/test_camera_motion.sh