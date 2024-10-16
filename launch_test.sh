export dbad=1025
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4
checkpoint_path=lavis/output/scoring_cls/blip2_vicuna_instruct_vicuna7b/train/b16_e20_lr0.0001_wd0.05_q32_f100_fb20_freezevit/checkpoint_latest.pth
log_path=./logs/scoring/test.log
bash ./run_scripts/scoring/test.sh $checkpoint_path > $log_path 2>&1 &
code $log_path

