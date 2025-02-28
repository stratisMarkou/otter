"""Base MLP model, for use within the transformer."""

import torch

from otter.models.architectures.transformer_output import SwinTransformerOutput


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
            + [
                torch.nn.Linear(
                    hidden_features,
                    output_features,
                )
            ]
        )

    def forward(
        self, x: SwinTransformerOutput | torch.Tensor
    ) -> SwinTransformerOutput | torch.Tensor:

        if isinstance(x, SwinTransformerOutput):
            pred = x.pred
        else:
            pred = x

        if pred.shape[-1] != self.in_features:
            raise ValueError(
                f"Input tensor has shape {pred.shape}, "
                f"expected {self.in_features}"
            )
        for layer in self.layers:
            pred = layer(pred)

        if isinstance(x, SwinTransformerOutput):
            x.pred = pred
            return x
        return pred
