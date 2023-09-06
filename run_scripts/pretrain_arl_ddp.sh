#!/bin/bash

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

#--- Multi-nodes training hyperparams ---
MIN_SIZE=1
MAX_SIZE=1
TRAINERS_PER_NODE=1
JOB_ID=arlddp
port=23456
HOST_NODE_ADDR="127.0.0.1:"${port}
master_addr="127.0.0.1"
RANK=0

# python -m torch.distributed.launch \
#             --master_port $port \
#             --nproc_per_node=$TRAINERS_PER_NODE \
#             --nnodes=${MAX_SIZE} \
#             --node_rank=$RANK \
#             --master_addr=${master_addr} \
#             main.py


torchrun main.py
