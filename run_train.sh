#! /bin/bash

GPU=0,1,2,3,4,5,6,7
DATA_DIR=./Data
EXP_DIR=./Exps
COMMON_TASK_NAME=try
RESUME_TRAIN=True
SAVE_CPT=True
N_EPOCH=100
TRAIN_BS=64
EVAL_BS=2
NUM_GPUS=8
GRAD_ACC_STEP=8
MODEL_STR=GIT

echo "---> ${MODEL_STR} Run"
CUDA_VISIBLE_DEVICES=${GPU} ./train_multi.sh ${NUM_GPUS} --resume_latest_cpt ${RESUME_TRAIN} --save_cpt_flag ${SAVE_CPT} \
    --data_dir ${DATA_DIR} --exp_dir ${EXP_DIR} --task_name ${COMMON_TASK_NAME} --num_train_epochs ${N_EPOCH} \
    --train_batch_size ${TRAIN_BS} --gradient_accumulation_steps ${GRAD_ACC_STEP} --eval_batch_size ${EVAL_BS} \
    --cpt_file_name ${MODEL_STR}
