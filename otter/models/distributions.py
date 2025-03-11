from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TypeVar

import torch
import torch.distributions as dist

T = TypeVar("T", bound="Distribution")


class Distribution(ABC):
    num_stats: int

    def __init__(self):
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def stack(self, dists: List[T], dim: int) -> T:
        pass

    @property
    @abstractmethod
    def num_stats(self) -> int:
        pass


class Delta(Distribution):
    num_stats: int = 1

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def sample(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def stack(dists: List[Delta], dim: int) -> Delta:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        return Delta(mean)


class Gaussian(Distribution):
    num_stats: int = 2

    def __init__(
        self,
        mean: torch.Tensor,
        unconstrained_stddev: torch.Tensor,
        eps: float = 1e-12,
    ):
        self.mean = mean
        self.unconstrained_stddev = unconstrained_stddev
        self.stddev = torch.nn.Softplus(unconstrained_stddev) + eps
        self.dist = dist.Normal(self.mean, self.stddev)

    def sample(self) -> torch.Tensor:
        return self.dist.rsample()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(x)

    @staticmethod
    def stack(dists: List[Gaussian], dim: int) -> Gaussian:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        stddev = torch.stack([dist.stddev for dist in dists], dim=dim)
        return Gaussian(mean, stddev)


def product_distribution(dists: List[T], dim: int) -> T:
    if not dists:
        raise ValueError("Cannot stack empty list of distributions")

    return dists[0].stack(dists, dim)
