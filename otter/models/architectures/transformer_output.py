from dataclasses import dataclass

import torch


@dataclass
class LoadBalancingLosses:
    importance_loss: torch.Tensor
    load_loss: torch.Tensor
