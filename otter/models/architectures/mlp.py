"""Base MLP model, for use within the transformer."""

import torch
from torch import nn


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_hidden_layers: int,
        output_features: int,
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
        )
        self.final_layer = nn.Linear(hidden_features, output_features)

    def __forward__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input tensor has shape {x.shape}, "
                f"expected {self.in_features}"
            )
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)
