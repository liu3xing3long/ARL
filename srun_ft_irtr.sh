
seed=0
GPUS=4
GPUS_PER_NODE=4
NODE_COUNT=1
PARTITION=MIA_LLM
JOB_NAME=ARL_ft_irtr
per_gpu_batchsize=4
finetune_embeddings=True
#load_path="result/task_pretrain_arl-seed0-from_downloaded_meter.ckpt/version_11/checkpoints/epoch=2-step=8952.ckpt"
load_path="result/task_pretrain_arl-seed0-from_downloaded_meter.ckpt/version_12/checkpoints/epoch=4-step=14920.ckpt"

# === IRTR ===
srun -n ${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
     num_gpus=${GPUS} num_nodes=${NODE_COUNT} ddp_port=25902 \
     task_finetune_irtr_roco get_recall_metric=False \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip16 text_roberta \
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
     task_finetune_irtr_roco get_recall_metric=True \
     per_gpu_batchsize=${per_gpu_batchsize} \
     clip16 text_roberta \
     image_size=288 \
     tokenizer=downloaded/roberta-base \
     finetune_embeddings=${finetune_embeddings} \
     test_only=True \
     load_path=${load_path}

