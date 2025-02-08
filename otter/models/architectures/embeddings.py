import einops
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from otter.models.architectures.utils import interpolate_bilinear_channel_last

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
        self.scales = torch.exp(log_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create Fourier embeddings from input tensor x.

        Arguments:
            x: input tensor of shape (...)

        Returns:
            output tensor of shape (..., 2 * num_scales)
        """

        # TODO: Check if we want to use float64 here like in Aurora
        x = x.double()

        scales = self.scales.to(x.device)

        # Compute the Fourier embeddings.
        prod = torch.einsum("...d, s->...ds", x, scales**-1.0)

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
        return super().forward(torch.tensor(hours).float())


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Construct a meshgrid of latitude and longitude coordinates.

    `torch.meshgrid(*tensors, indexing="xy")` gives same behavior as calling
    `numpy.meshgrid(*arrays, indexing="ij")`::

        lat = torch.tensor([1, 2, 3])
        lon = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
        grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
        grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])

    Args:
        lat (torch.Tensor): Vector of latitudes.
        lon (torch.Tensor): Vector of longitudes.

    Returns:
        torch.Tensor: Meshgrid of shape `(2, len(lat), len(lon))`.
    """
    assert lat.dim() == 1
    assert lon.dim() == 1

    grid_x, grid_y = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack((grid_x, grid_y), dim=0)
    grid = grid.permute(0, 2, 1)

    return grid


class SpatialEmbedding(nn.Module):
    """Position embedding class for 2-D lat-lon data"""

    def __init__(
        self,
        token_dimension: int,
        height: int,
        width: int,
        num_scales: int,
        patch_size: int | tuple[int, ...],
        min_scale: float = SPATIAL_EMBEDDING_DEFAULT_MIN_SCALE,
        max_scale: float = SPATIAL_EMBEDDING_DEFAULT_MAX_SCALE,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.height = height
        self.width = width
        if isinstance(patch_size, int):
            self.patch_size = tuple((patch_size, patch_size))
        else:
            self.patch_size = patch_size

        self.spatial_emb_h = FourierExpansionEmbedding(
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales // 2,
        )

        self.spatial_emb_w = FourierExpansionEmbedding(
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales // 2,
        )

        self.linear = nn.Linear(2 * num_scales, token_dimension)

    def _get_lat_lon_mebeddings(
        self, grid_h: torch.Tensor, grid_w: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to compute the lat-lon embeddings"""
        B = grid_h.shape[0]

        # Compute Fourier embeddings for latitude and longitude.
        encode_h = self.spatial_emb_h(grid_h.reshape(B, -1))  # (B, L, D/2)
        encode_w = self.spatial_emb_w(grid_w.reshape(B, -1))  # (B, L, D/2)

        # Concatenate lat and lon embeddings.
        spatial_encode = torch.cat((encode_h, encode_w), dim=-1)  # (B, L, D)
        return spatial_encode

    def forward(
        self,
        x: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:

        # Create latitude-longitude grid.
        grid = lat_lon_meshgrid(lat, lon)
        grid = grid[None]

        # Interpolate grid to target dimensions.
        grid = interpolate_bilinear_channel_last(
            grid.permute(0, 2, 3, 1),
            height=self.height,
            width=self.width,
        ).permute(0, 3, 1, 2)

        # Ensure embedding dimension is valid.
        assert (
            self.num_scales % 4 == 0
        ), f"Embedding dim {self.num_scales} not divisible by 4"
        assert grid.dim() == 4  # (1, 2, lat, lon)

        # Create a 2D mesh with pooled values.
        grid_h = F.avg_pool2d(grid[:, 0], self.patch_size)
        grid_w = F.avg_pool2d(grid[:, 1], self.patch_size)

        # Get lat and lon embeddings.
        spatial_encode = self._get_lat_lon_mebeddings(grid_h, grid_w)

        # Project to token dimension.
        spatial_encode = self.linear(spatial_encode.to(x.device))[
            :, None, ...
        ]  # (B, 1, L, D_emb)

        # Rearrange in grid shape.
        spatial_encode = einops.rearrange(
            spatial_encode, "b 1 (h w) d -> b h w d", h=x.shape[1]
        )

        return x + spatial_encode
