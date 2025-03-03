"""Base MLP model, for use within the transformer."""

from typing import Optional

import torch

from otter.models.architectures.transformer_output import LoadBalancingLosses


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        output_features: int,
        hidden_features: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.in_features = in_features
        if num_hidden_layers < 1:
            raise ValueError("Number of hidden layers must be at least 1.")

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features if i == 0 else hidden_features,
                        hidden_features,
                    ),
                    torch.nn.GELU(),
                )
                for i in range(num_hidden_layers)
            ]
            + [
                torch.nn.Linear(
                    hidden_features,
                    output_features,
                )
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input tensor has shape {x.shape}, "
                f"expected {self.in_features}"
            )
        for layer in self.layers:
            x = layer(x)

        return x, None
