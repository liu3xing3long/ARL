seed=42
num_gpus=1
per_gpu_batchsize=16
finetune_embeddings=True
#load_path=<Path to The Pre-trained Model>
load_path="result/task_pretrain_arl-seed0-from_downloaded_meter.ckpt/version_11/checkpoints/epoch=2-step=8952.ckpt"

# === VQA ===
python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path}

python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop

python main.py with seed=${seed} data_root=data/finetune_arrows_umls/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 finetune_embeddings=${finetune_embeddings} \
 load_path=${load_path} \
 clip_resizedcrop
