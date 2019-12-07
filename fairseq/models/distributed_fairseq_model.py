# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect

import torch.nn as nn

from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel
from fairseq.models import BaseFairseqModel


def DistributedFairseqModel(args, model):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """
    # determine which DDP class to extend
    assert isinstance(model, nn.Module)
    if args.ddp_backend == 'c10d': ## ddp backend 默认为 c10d
        #init_process_group后，相当于整个机器环境都处于process_group中, torch.distributed相当于一个全局的环境变量
        #可通过torch.distributed来获取process_group的相关信息
        #init_process_group后，可直接使用这个类对神经网络进行wrap实现数据并行, 见evernote笔记对于此类的解读
        #现在的理解：基于/path-to-torch/torch/distributed/distributed_10.py
        #nn.parallel.DistributedDataParallel，相当于在model包了一层wrapper，便于参数(graidents)在多个机器的同步和更新
        #LegacyDistributedDataParallel是fairseq的作者们自己实现的简单版本的DistributedFairseqModel, 可以看它帮助理解
        #实现思路：为每个参数的graident进行register_hook，hook函数即为实现多GPU卡之间的同步(后续写一篇单独的文章讲这个)
        #现有理解下每个机器仍然需要独自进行：
        #1. 如果要恢复checkpoint, 那么每个GPU要自己读入对应的checkpoint ?? 这个待确定
        #2. 每个机器要构造自己的data_iterator根据distributed_rank去iterate自己的那份数据
        #3. 不同GPU之间唯一相互同步的就是graident的，其他都靠自己！！
        ddp_class = nn.parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_cap_mb,
        )
        # Maintain backward compatibility
        if 'check_reduction' in inspect.getargspec(ddp_class)[0]:
            init_kwargs['check_reduction'] = True
        if 'find_unused_parameters' in inspect.getargspec(ddp_class)[0]:
            init_kwargs['find_unused_parameters'] = args.find_unused_parameters
    elif args.ddp_backend == 'no_c10d':
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict(
            module=model,
            world_size=args.distributed_world_size,
            buffer_size=2**28,
        )
    else:
        raise ValueError('Unknown --ddp-backend: ' + args.ddp_backend)

    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedFairseqModel(**init_kwargs)
