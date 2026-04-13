#!/bin/bash

# Usage: bash run_td3.sh <SEED>
# SEED=$1

# if [ -z "$SEED" ]; then
#   echo "Error: Must provide seed as first argument."
#   exit 1
# fi


# SAVE_PREFIX="/home/yuxin/infgen"
# EXP_NAME="0511_CLRL_SCGEN_last_step_col"
# SAVE_PATH="${SAVE_PREFIX}/${EXP_NAME}_seed${SEED}"
# mkdir -p ${SAVE_PATH}

python ../train/train_td3.py \
--exp_name=0521_OLRL_SCGEN_4_mode \
--wandb_project=scgen \
--wandb_team=drivingforce \
--seed=300 \
--data_dir=/bigdata/yuxin/mixed_Waymo_BMT_scenarionet_waymo_training_500_4_mode/ \
--eval_data_dir=/bigdata/yuxin/scenarionet_waymo_validation_100 \
--save_path=/home/yuxin/infgen/0521_OLRL_SCGEN_4_mode_seed300 \
--training_step=5000000 \
--lr=1e-4 \
--eval_freq=100000 \
--horizon=100 \
--eval_horizon=100 \
--num_eval_envs=1 \
--eval_ep=100 \
--wandb \
  > ${EXP_NAME}_seed${SEED}.log 2>&1