# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq import registry
from fairseq.optim.fairseq_optimizer import FairseqOptimizer
from fairseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from fairseq.optim.bmuf import FairseqBMUF  # noqa


__all__ = [
    'FairseqOptimizer',
    'FP16Optimizer',
    'MemoryEfficientFP16Optimizer',
]

# 根据registry.py，由于各个optimizer都没有build_optimizer函数，
# 因此在setup_registry函数中的build_x函数中，返回的就是对应class的构造函数
# 也就是根据args.optimizer的值，这个值对应的optimizer class(比如值adam对应FairseqAdam类)没有build_optimizer函数，
# 于是registry.setup_registry函数中的build_x函数就会返回对应class的构造函数，
# 也就是说build_optimizer就是args.optimizer对应的optimizer的构造函数
build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    '--optimizer',
    base_class=FairseqOptimizer,
    default='nag',
)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.' + module)
