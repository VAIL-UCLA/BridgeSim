#!/bin/bash

# Usage: bash run_td3.sh <SEED>
SEED=$1

if [ -z "$SEED" ]; then
  echo "Error: Must provide seed as first argument."
  exit 1
fi


SAVE_PREFIX="/home/yuxin/infgen"
EXP_NAME="0511_CLRL_SCGEN_last_step_col"
SAVE_PATH="${SAVE_PREFIX}/${EXP_NAME}_seed${SEED}"
mkdir -p ${SAVE_PATH}



python ../train/train_td3.py \
  --exp_name=${EXP_NAME} \
  --wandb_project='scgen' \
  --wandb_team='drivingforce' \
  --seed=${SEED} \
  --data_dir=/bigdata/yuxin/scenarionet_waymo_training_500 \
  --eval_data_dir=/bigdata/yuxin/scenarionet_waymo_validation_100 \
  --save_path=${SAVE_PATH} \
  --training_step=2000000 \
  --lr=1e-4 \
  --eval_freq=50000 \
  --horizon=100 \
  --model_name=None \
  --eval_horizon=100 \
  --closed_loop \
  --source_data=/bigdata/yuxin/scenarionet_waymo_training_500 \
  --closed_loop_generator=SCGEN \
  --num_eval_envs=1 \
  --eval_ep=100 \
  --wandb \
  > ${EXP_NAME}_seed${SEED}.log 2>&1

