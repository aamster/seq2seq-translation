import torch
from torch.nn.parallel import DistributedDataParallel


def model_isinstance(m, type):
    """Unwrap any DDP or OptimizedModule wrappers to get the original model."""
    # Unwrap DDP
    if isinstance(m, DistributedDataParallel):
        m = m.module

    # Unwrap torch.compile() wrapper
    if isinstance(m, torch._dynamo.eval_frame.OptimizedModule):
        m = m._orig_mod

    return isinstance(m, type)


def unwrap_model(m):
    """Unwrap any DDP or OptimizedModule wrappers to get the original model."""
    if isinstance(m, DistributedDataParallel):
        m = m.module

    # Unwrap torch.compile() wrapper
    if isinstance(m, torch._dynamo.eval_frame.OptimizedModule):
        m = m._orig_mod

    return m
