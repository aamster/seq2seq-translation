import os
from datetime import timedelta

import torch
from torch.distributed import init_process_group, destroy_process_group


class DistributedContextManager:
    def __init__(self, backend="nccl"):
        self._backend = backend
        self._ddp_rank = None
        self._ddp_local_rank = None

    def __enter__(self):
        init_process_group(backend=self._backend, timeout=timedelta(minutes=30))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self._ddp_rank = ddp_rank
        self._ddp_local_rank = ddp_local_rank
        return self

    @property
    def is_master_process(self) -> bool:
        return self._ddp_rank == 0

    @property
    def seed_offset(self):
        return self._ddp_rank

    @property
    def ddp_local_rank(self):
        return self._ddp_local_rank

    @property
    def rank(self):
        return self._ddp_rank

    def __exit__(self, exc_type, exc_val, exc_tb):
        destroy_process_group()


class SingleProcessContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def is_master_process(self) -> bool:
        return True

    @property
    def seed_offset(self):
        return 0

    @property
    def ddp_local_rank(self):
        return "main"

    @property
    def rank(self):
        return "main"


def is_master_process():
    return (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ) == 0


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
