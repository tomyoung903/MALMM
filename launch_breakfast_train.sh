export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1 #,1,2,3,4,5
export dbad=1025
bash ./run_scripts/breakfast/train.sh > ./logs/breakfast/train.log 2>&1 &
code ./logs/breakfast/train.log