from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Optional,
)

import torch


class BaseScaleScheduler(ABC):
    @abstractmethod
    def sample_scale(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class PowerScaleScheduler(BaseScaleScheduler):
    def __init__(
        self,
        num_scales: int,
        scale_probs: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.scales = torch.pow(
            2, torch.arange(num_scales, 0, -1, dtype=torch.int)
        )
        if scale_probs is None:
            self.scale_probs = torch.ones(
                len(self.scales), dtype=torch.float32
            )
        else:
            self.scale_probs = torch.tensor(scale_probs, dtype=torch.float32)

        assert len(self.scales) == len(self.scale_probs), (
            f"Mismatch between number of scales {len(self.scales)} "
            f"and number of associated probabilities {len(self.scale_probs)}"
        )

    def sample_scale(self) -> int:
        # Sample according to scale probability
        return int(self.scales[torch.multinomial(self.scale_probs, 1)].item())

    def __len__(self) -> int:
        return len(self.scales)

    def to(self, device: str, non_blocking: bool = False) -> None:
        self.scales = self.scales.to(device, non_blocking=non_blocking)
