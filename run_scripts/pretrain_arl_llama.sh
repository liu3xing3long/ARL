python main.py with data_root=data/pretrain_arrows_umls/ \
 num_gpus=1 num_nodes=1 \
 task_pretrain_arl_llama \
 per_gpu_batchsize=16 \
 clip16 text_biomedllama2at7b \
 image_size=288 max_text_len=64 max_num_ents=24
 