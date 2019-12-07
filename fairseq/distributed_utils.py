# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import socket
import subprocess
import warnings

import torch
import torch.distributed as dist

from fairseq import utils


def is_master(args):
    return args.distributed_rank == 0


def infer_init_method(args):
    if args.distributed_init_method is not None:
        return

    # support torch.distributed.launch 多机训练 
    # torch.distributed.launch 位置：/path-to-torch/torch/distributed/launch.py
    # torch.distributed.launch 会根据输入的nproc_per_node数目循环调用train.py来创建子进程 
    # 每个进程使用的本节点的GPU id作为local_rank参数，传入运行代码
    # 环境变量RANK中存放，当前GPU在所有节点的所有GPU中的ID
    # 环境变量WORLD_SIZE 存放，多机训练时使用的所有的GPU的数目：nnodes*nproc_per_node
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        args.distributed_init_method = 'env://'
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_STEP_NODELIST')
        if node_list is None:
            node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port,
                )
                nnodes = int(os.environ.get('SLURM_NNODES'))
                ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get('SLURM_NTASKS'))
                    nnodes = int(os.environ.get('SLURM_NNODES'))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert args.distributed_world_size % nnodes == 0
                    gpus_per_node = args.distributed_world_size // nnodes
                    node_id = int(os.environ.get('SLURM_NODEID'))
                    args.distributed_rank = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == args.distributed_world_size // nnodes
                    args.distributed_no_spawn = True
                    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                    args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(args): ## 单机多卡和多机多卡训练都会调用这个函数
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    if torch.distributed.is_initialized():
        warnings.warn('Distributed is already initialized, cannot initialize twice!')
    else:
        print('| distributed init (rank {}): {}'.format(
            args.distributed_rank, args.distributed_init_method), flush=True)
        dist.init_process_group(
            backend=args.distributed_backend, ## 设置什么样的backend，用于操作分布式的数据传输和通信等工作
            init_method=args.distributed_init_method, ## 设置 distributed_init_method 为 'env://'
            world_size=args.distributed_world_size, ## 所有使用的GPU的数量：nnodes * nproc_per_node
            rank=args.distributed_rank, ## 告诉backend，当前gpu在所有gpu上的id
        )
        print('| initialized host {} as rank {}'.format(
            socket.gethostname(), args.distributed_rank), flush=True)

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            ##inplace操作，把各个gpu上的tensor进行合并。默认设置为同步、加法
            ##相当于此处把各个GPU进行一次同步
            dist.all_reduce(torch.zeros(1).cuda()) 
        else:
            dist.all_reduce(torch.zeros(1))

        suppress_output(is_master(args)) ##重置print函数，只在master GPU即distributed_rank=0的GPU上进行打印。

    args.distributed_rank = torch.distributed.get_rank() ##获取当前GPU在所有GPU中的id
    return args.distributed_rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256

    cpu_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    cpu_buffer[1] = enc_size % 255
    cpu_buffer[2 : enc_size + 2] = torch.ByteTensor(list(enc))
    start = rank * max_size
    size = enc_size + 2
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = (255 * utils.item(out_buffer[0])) + utils.item(out_buffer[1])
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[2 : size + 2].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )
