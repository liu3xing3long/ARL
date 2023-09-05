import copy
import os
import resource

import torch
import pytorch_lightning as pl

from arl.config import ex
from arl.datamodules.multitask_datamodule import MTDataModule
from arl.modules import ARLTransformerSS

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def fake_init_dist_slurm(backend='nccl', port=29500):
    # if mp.get_start_method(allow_none=True) != 'spawn':
    #     mp.set_start_method('spawn')

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
    # dist.init_process_group(backend='nccl')
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # return rank, world_size


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ================================
    print('->' * 10 + 'final config' + '->' * 10)
    for k, v in _config.items():
        print(fr'{k} -> {v}')
    print('<-' * 10 + 'final config' + '<-' * 10)
    # ================================

    # ================================
    fake_init_dist_slurm()
    torch.set_float32_matmul_precision('high')

    os.environ['LOCAL_RANK'] = str(_config['local_rank'])
    print(fr'num_gpus {_config["num_gpus"]}, numnodes {_config["num_nodes"]}')

    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    else:
        rank = -1

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = -1

    if 'LOCAL_RANK' in os.environ:
        local_rank = os.environ['LOCAL_RANK']
    else:
        local_rank = -1

    rank_num_gpus = torch.cuda.device_count()
    print(fr'dist settings: local_rank {local_rank}, rank/worldsize : {rank} / {world_size}, rank-gpus {rank_num_gpus}')
    # ================================

    # Data modules
    dm = MTDataModule(_config, dist=True)

    # Module
    model = ARLTransformerSS(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    # wb_logger = pl.loggers.WandbLogger(
    #     project="arl-finetune" if "finetune" in exp_name else "arl-pretrain",
    #     name=run_name
    # )
    # loggers = [tb_logger, wb_logger]
    loggers = [tb_logger, ]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 1000

    print(fr'train settings: num_gpus {num_gpus}, num_workers {_config["num_workers"]}')

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        enable_model_summary=True,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        if "pretrain" not in _config["exp_name"]:
            trainer.test(ckpt_path="best" if "irtr" not in _config["exp_name"] else None, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
