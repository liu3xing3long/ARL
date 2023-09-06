import copy
import os
import resource
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from arl.config import ex
from arl.datamodules.multitask_datamodule import MTDataModule
from arl.modules import ARLTransformerSS

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

######################################################
g_rank = -1
g_world_size = 0
best_loss = -100

######################################################
def _log_msg(strmsg="\n"):
    global g_rank
    if g_rank == 0:
        print(strmsg)

######################################################
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def average_variable(var):
    size = float(dist.get_world_size())
    dist.all_reduce(var, op=dist.ReduceOp.SUM)
    var /= size
    return var

def init_dist_slurm(backend='nccl', port=29500):
    # if mp.get_start_method(allow_none=True) != 'spawn':
    #    mp.set_start_method('spawn')

    if 'SLURM_PROCID' not in os.environ:
        return 
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')

    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = os.environ['LOCAL_RANK'] 
    print(fr'dist settings: local_rank {local_rank}, rank {rank}, worldsize {world_size}, gpus {num_gpus}')

    return rank, world_size

def init_dist(backend='nccl', port=29500):
    # if mp.get_start_method(allow_none=True) is None:
    #    mp.set_start_method('spawn')

    master_ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    local_rank = os.environ['LOCAL_RANK'] 

    torch.cuda.set_device(eval(local_rank) % num_gpus)
    dist.init_process_group(backend=backend)

    print(fr'dist settings: local_rank {local_rank}, rank {rank}, worldsize {world_size}, gpus {num_gpus}')
    return rank, world_size

######################################################
def train(model, dataloader, epoch):
    model.train()
    for batch_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        model.module.training_step(batch, global_step=epoch * len(dataloader) + batch_index)  
    model.module.on_train_epoch_end()

######################################################
def validate(model, dataloader, epoch):
    model.val()
    return -100.0

######################################################
def test(model, dataloader, epoch):
    model.val()
    pass

######################################################
@ex.main
def main(_config):
    _log_msg('running main')
    
    global best_loss, g_rank, g_world_size

    # ================================  
    _config = copy.deepcopy(_config)
    # print(_config)
    
    # ================================
    is_cluster = False
    local_rank = _config['local_rank']

    os.environ['LOCAL_RANK'] = fr'{local_rank}'
    if is_cluster:
        rank, world_size = init_dist_slurm()
    else:
        rank, world_size = init_dist()

    g_rank = rank
    g_world_size = world_size
    _log_msg(fr"dist settings:{local_rank}, {g_rank} / {g_world_size}")

    torch.set_float32_matmul_precision('high')

    # ================================
    # Loggers
    log_dir = _config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    full_log_path = fr'{log_dir}/{run_name}'
    os.makedirs(full_log_path, exist_ok=True)

    # Data modules
    datamodule = MTDataModule(_config, dist=True)

    # Module
    model = ARLTransformerSS(_config, outputpath=full_log_path)

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1
    max_epochs = _config["max_epoch"] if max_steps == -1 else 1000

    _log_msg(fr'train settings: num_gpus {num_gpus}, num_workers {_config["num_workers"]}')
    _log_msg(fr'train settings: max_epochs {max_epochs}, max_steps {max_steps}, '
             fr'valcheck_interval {_config["val_check_interval"]}')

    ##############################
    model = model.cuda()
    model.configure_optimizers(len(datamodule.train_dataset), max_epochs, grad_steps)

    broadcast_params(model)
    ddp_model = DistributedDataParallel(model, device_ids=[g_rank], output_device=g_rank)
    pytorch_total_params = sum(p.numel() for p in ddp_model.parameters())
    _log_msg("Total number of params = {}".format(pytorch_total_params))

    ##############################
    # run train and validate
    start_epoch = 0
    for epoch in range(start_epoch, max_epochs + 1):

        ##################################
        if datamodule.train_sampler is not None:
            datamodule.train_sampler.set_epoch(epoch)
        # Train for one epoch
        train(ddp_model, datamodule.train_dataloader(), epoch)

        ##################################
        if datamodule.val_sampler is not None:
            datamodule.val_sampler.set_epoch(epoch)
        # Evaluate on validation set
        val_loss = validate(ddp_model, datamodule.val_dataloader(), epoch)


if __name__ == '__main__':
    ex.run(
        named_configs=(
            'task_pretrain_arl', 
            'clip16', 
            'text_roberta',
        ),
        config_updates={
            "data_root": "data/pretrain_arrows_umls/",
            "num_gpus": 1, 
            "num_nodes": 1,
            "per_gpu_batchsize": 16,
            "image_size": 288, 
            "max_text_len": 64, 
            "max_num_ents": 24,
            "tokenizer": "downloaded/roberta-base",
            "load_path": "downloaded/meter.ckpt",
        },
    )
