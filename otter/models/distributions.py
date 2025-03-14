from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from typing_extensions import Self

import torch
import torch.distributions as dist


class Distribution(ABC):
    num_stats: int

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def stack(cls, dists: List[Self], dim: int) -> Self:
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

    @classmethod
    def stack(cls, dists: List[Self], dim: int) -> Self:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        return cls(mean)

    def __add__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean + other)

    def __mul__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean * other)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(1 + torch.exp(x))


def inv_softplus(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.exp(x) - 1 + eps)


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
        self.stddev = softplus(unconstrained_stddev) + eps
        self.dist = dist.Normal(self.mean, self.stddev)  # type: ignore

    def sample(self) -> torch.Tensor:
        return self.dist.rsample()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob: torch.Tensor = self.dist.log_prob(x)  # type: ignore
        return log_prob

    @classmethod
    def stack(cls, dists: List[Self], dim: int) -> Self:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        stddev = torch.stack([dist.stddev for dist in dists], dim=dim)
        return cls(mean, inv_softplus(stddev))

    def __add__(self, other: torch.Tensor) -> Gaussian:
        return Gaussian(self.mean + other, self.unconstrained_stddev)

    def __mul__(self, other: torch.Tensor) -> Gaussian:
        unconstrained_stddev = inv_softplus(self.stddev * other)
        return Gaussian(self.mean * other, unconstrained_stddev)


def product_distribution(dists: List[Distribution], dim: int) -> Distribution:
    if not dists:
        raise ValueError("Cannot stack empty list of distributions")

    return dists[0].stack(dists, dim)
