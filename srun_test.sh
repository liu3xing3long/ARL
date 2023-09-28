#!/bin/bash

seed=0
GPUS=1
GPUS_PER_NODE=1
NODE_COUNT=1
PARTITION=MIA_LLM
JOB_NAME=ARL_test_vqa
per_gpu_batchsize=1
finetune_embeddings=True
load_path="result/task_finetune_vqa_vqa_rad-seed0-from_result_task_pretrain_arl-seed0-from_downloaded_meter.ckpt_version_11_checkpoints_epoch=2-step=8952.ckpt/version_0/checkpoints/epoch=39-step=1000.ckpt"
#load_path="result/task_pretrain_arl-seed0-from_downloaded_meter.ckpt/version_11/checkpoints/epoch=2-step=8952.ckpt"

# === VQA ===
srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} \
     task_finetune_vqa_vqa_rad \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip16 text_roberta \
     image_size=384 \
     test_only=1 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     load_path=${load_path}

