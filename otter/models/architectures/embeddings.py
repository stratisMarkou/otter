import numpy as np
import numpy.typing as npt
import torch
from torch import nn

TEMPORAL_EMBEDDING_DEFAULT_MIN_HOUR_SCALE = 3.0
TEMPORAL_EMBEDDING_DEFAULT_MAX_HOUR_SCALE = 8760.0
REFERENCE_DATETIME = np.datetime64("1979-01-01T00:00:00")


class PositionEmbedding(nn.Module):

    def __init__(
        self,
        token_dimension: int,
        height: int,
        width: int,
        init_scale: float = 1e-1,
    ):
        super().__init__()

        self.embeddings = nn.Parameter(
            init_scale * torch.randn(height, width, token_dimension),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input tensor.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        return x + self.embeddings[None, :, :, :]


class FourierExpansionEmbedding(nn.Module):

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        num_scales: int,
    ) -> None:
        super().__init__()

        # min_scale = torch.tensor(np.array(min_scale))
        # max_scale = torch.tensor(np.array(max_scale))

        # Set up scales in log space.
        log_scales = torch.linspace(
            torch.log(torch.tensor(np.array(min_scale))),
            torch.log(torch.tensor(np.array(max_scale))),
            num_scales,
        )
        self.scales = torch.exp(log_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create Fourier embeddings from input tensor x.

        Arguments:
            x: input tensor of shape (...)

        Returns:
            output tensor of shape (..., 2 * num_scales)
        """

        scales = self.scales.to(x.device)

        # Compute the Fourier embeddings.
        prod = torch.einsum("...d, s->...ds", x, scales**-1.0)

        sin = torch.sin(2 * torch.pi * prod)  # (..., num_scales)
        cos = torch.cos(2 * torch.pi * prod)  # (..., num_scales)

        return torch.cat([sin, cos], dim=-1)  # (..., 2 * num_scales)


def _get_hours_from_reference_time(
    x: npt.NDArray[np.datetime64],
) -> npt.NDArray[np.int32]:
    return (x - REFERENCE_DATETIME).astype("timedelta64[h]").astype(int)


class TimeEmbedding(FourierExpansionEmbedding):
    """Fourier embedding class that works with"""

    def __init__(
        self,
        num_scales: int,
        min_scale: float = TEMPORAL_EMBEDDING_DEFAULT_MIN_HOUR_SCALE,
        max_scale: float = TEMPORAL_EMBEDDING_DEFAULT_MAX_HOUR_SCALE,
    ) -> None:
        super().__init__(min_scale, max_scale, num_scales)

    def forward(
        self,
        x: npt.NDArray[np.datetime64],  # type: ignore
    ) -> torch.Tensor:
        hours = _get_hours_from_reference_time(x)
        return super().forward(torch.tensor(hours).float())
