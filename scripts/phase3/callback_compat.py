"""Import-only compatibility for unused FSDP2 callback symbols on torch 2.5."""
from __future__ import annotations

import torch.distributed.fsdp as fsdp


if not hasattr(fsdp, "FSDPModule"):
    class FSDPModule:
        pass

    fsdp.FSDPModule = FSDPModule
