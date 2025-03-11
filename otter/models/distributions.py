from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TypeVar

import torch
import torch.distributions as dist

T = TypeVar("T", bound="Distribution")


class Distribution(ABC):
    num_stats: int

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def stack(dists: List[T], dim: int) -> T:
        pass

    @abstractmethod
    def __add__(self, other: torch.Tensor) -> Distribution:
        pass

    @abstractmethod
    def __mul__(self, other: torch.Tensor) -> Distribution:
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

    def __add__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean + other)

    def __mul__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean * other)


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
        self.stddev = torch.exp(unconstrained_stddev) + eps
        self.dist = dist.Normal(self.mean, self.stddev)

    def sample(self) -> torch.Tensor:
        return self.dist.rsample()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob: torch.Tensor = self.dist.log_prob(x)
        return log_prob

    @staticmethod
    def stack(dists: List[Gaussian], dim: int) -> Gaussian:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        stddev = torch.stack([dist.stddev for dist in dists], dim=dim)
        return Gaussian(mean, stddev)

    def __add__(self, other: torch.Tensor) -> Gaussian:
        return Gaussian(self.mean + other, self.unconstrained_stddev)

    def __mul__(self, other: torch.Tensor) -> Gaussian:
        unconstrained_stddev = torch.log(self.stddev * other)
        return Gaussian(self.mean * other, unconstrained_stddev)


def product_distribution(dists: List[T], dim: int) -> T:
    if not dists:
        raise ValueError("Cannot stack empty list of distributions")

    return dists[0].stack(dists, dim)
