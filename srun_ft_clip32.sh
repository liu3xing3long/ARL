#!/bin/bash

seed=0
GPUS=1
GPUS_PER_NODE=1
NODE_COUNT=1
PARTITION=MIA_LLM
JOB_NAME=ARL_ft_vqa
per_gpu_batchsize=16
finetune_embeddings=True
load_path="result/task_pretrain_arl-seed0-from_/version_0/checkpoints/epoch=8-step=26856.ckpt"

# === VQA ===
srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_vqa_vqa_rad \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip32 text_roberta \
     image_size=384 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     load_path=${load_path}

srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_vqa_slack \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip32 text_roberta \
     image_size=384 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     load_path=${load_path} \
     clip_resizedcrop

srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_vqa_medvqa_2019 \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip32 text_roberta \
     image_size=384 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     load_path=${load_path} \
     clip_resizedcrop

# === CLS ===
srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_cls_melinda_p_meth_label \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip32 text_roberta \
     image_size=384 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     load_path=${load_path} \
     clip_resizedcrop

# === IRTR ===
srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_irtr_roco get_recall_metric=True \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip32 text_roberta \
     image_size=288 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     test_only=True \
     load_path=${load_path}





