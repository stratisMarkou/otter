from typing import Optional

import einops
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

TEMPORAL_EMBEDDING_DEFAULT_MIN_HOUR_SCALE = 3.0
TEMPORAL_EMBEDDING_DEFAULT_MAX_HOUR_SCALE = 8760.0
REFERENCE_DATETIME = np.datetime64("1979-01-01T00:00:00")
SPATIAL_EMBEDDING_DEFAULT_MIN_SCALE = 0.1
SPATIAL_EMBEDDING_DEFAULT_MAX_SCALE = 720


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

        # Set up scales in log space.
        log_scales = torch.linspace(
            torch.log(torch.tensor(np.array(min_scale))),
            torch.log(torch.tensor(np.array(max_scale))),
            num_scales,
        )
        scales = torch.exp(log_scales)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create Fourier embeddings from input tensor x.

        Arguments:
            x: input tensor of shape (...)

        Returns:
            output tensor of shape (..., 2 * num_scales)
        """

        # TODO: Check if we want to use float64 here like in Aurora
        x = x.double()

        # Compute the Fourier embeddings.
        prod = torch.einsum("...d, s->...ds", x, self.scales**-1.0)

        sin = torch.sin(2 * torch.pi * prod)  # (..., num_scales)
        cos = torch.cos(2 * torch.pi * prod)  # (..., num_scales)

        return torch.cat([sin, cos], dim=-1).float()  # (..., 2 * num_scales)


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
        return super().forward(
            torch.tensor(hours, device=self.scales.device).float()
        )


class SpatialEmbedding(FourierExpansionEmbedding):
    """Spatial position embedding based on Fourier Embeddings"""

    def __init__(
        self,
        num_scales: int,
        min_scale: float = SPATIAL_EMBEDDING_DEFAULT_MIN_SCALE,
        max_scale: float = SPATIAL_EMBEDDING_DEFAULT_MAX_SCALE,
        token_dimension: Optional[int] = None,
    ) -> None:
        super().__init__(min_scale, max_scale, num_scales // 2)

        if token_dimension is not None:
            self.linear = nn.Linear(2 * num_scales, token_dimension)

        self.num_scales = num_scales

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add Fourier-based positional embeddings to the input tensor.

        Arguments:
            x (torch.Tensor): Input tensor of shape (B, H, W, D)

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, D_final)
        """

        # Get lat-lon dimensions from tensor
        lat_dim, lon_dim = x.shape[1], x.shape[2]

        # Set up lat-lon coordinates
        lat = torch.linspace(-90, 90, lat_dim, device=x.device)
        lon = torch.linspace(
            0, 360 - (360 / lon_dim), lon_dim, device=x.device
        )

        # Get lat and lon embeddings.

        lat = super().forward(lat)  # (B, lat_dim, num_scales//2)
        lon = super().forward(lon)  # (B, lon_dim, num_scales//2)

        # Broadcast
        lat = einops.repeat(lat, "l d -> l L d", L=lon_dim)
        lon = einops.repeat(lon, "L d -> l L d", l=lat_dim)
        spatial_embedding = torch.cat([lat, lon], dim=-1)
        spatial_embedding = einops.repeat(
            spatial_embedding, "h w d -> b h w d", b=x.shape[0]
        )

        spatial_embedding = self.linear(spatial_embedding)
        return x + spatial_embedding
