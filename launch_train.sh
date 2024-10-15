#!/bin/bash
# always change the log file name
logfile=./logs/scoring/train_lr_1e-7_unfreeze_vit_vit_fp32.log
# rename the old output dir if it's a run with the same parameters
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export dbad=1025
bash ./run_scripts/scoring/train.sh > $logfile 2>&1 &
sleep 10
code $logfile