#!/bin/bash

GPUS=8
GPUS_PER_NODE=8
NODE_COUNT=1
PARTITION=MIA_LLM
JOB_NAME=ARLpretrain

srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with data_root=data/pretrain_arrows_umls/ \
    num_gpus=${GPUS_PER_NODE} num_nodes=${NODE_COUNT} \
    task_pretrain_arl \
    per_gpu_batchsize=16 batch_size=256 num_workers=16 \
    clip16 text_roberta \
    image_size=288 max_text_len=64 max_num_ents=24 \
    tokenizer=downloaded/roberta-base \
    load_path=downloaded/meter.ckpt

#        image_size=288 max_text_len=64 max_num_ents=24 \
