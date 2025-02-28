from dataclasses import (
    dataclass,
    field,
)
from typing import Dict

import torch


@dataclass
class SwinTransformerOutput:
    pred: torch.Tensor
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
