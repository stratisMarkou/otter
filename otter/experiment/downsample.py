import math

import torch
import torch.nn.functional as F


def downsample(original: torch.Tensor, divisor: int) -> torch.Tensor:
    """
    Downsample a batch of ERA5 data by a factor of `divisor`.
    The input tensor is expected to have the shape (B, Lat, Lon, Time, Var),
    where
        B is the batch size,
        Lat and Lon are the spatial dimensions,
        Time is the time dimension,
        Var is the variable/channel dimension.

    Note that the output tensor will have the same shape as the input tensor,
    but "pixelated" by the downsampling factor.

    Args:
        original: The input tensor.
        divisor: The downsampling factor, by which the spatial dimensions
            will be divided.
    """
    if divisor == 1:
        # No downsampling needed
        return original

    batch, lat, lon, time, var = original.shape

    lat_down = math.ceil(lat / divisor)
    lon_down = math.ceil(lon / divisor)

    if lat_down <= 1 or lon_down <= 1:
        # We have reached the lowest resolution, which is just the mean
        # of the entire grid, and can speed up the computation.
        mean = original.mean(dim=(1, 2), keepdim=True)
        return mean.expand(original.shape)

    x = original.permute(0, 3, 4, 1, 2)  # B, Time, Var, Lat, Lon
    # Merge Var and time, because the avg_pool functions require at most 4 dims
    x = x.reshape(batch, var * time, lat, lon)

    x = F.adaptive_avg_pool2d(
        x, (lat_down, lon_down)
    )  # B, Var*Time, Lat_down, Lon_down

    # Interpolate back up to the original shape
    x = F.interpolate(
        x, size=(lat, lon), mode="nearest"
    )  # B, Var*Time, Lat, Lon

    # Re-expand the Time dimension
    x = x.reshape(batch, time, var, lat, lon)
    return x.permute(0, 3, 4, 1, 2)  # B, Lat, Lon, Time, Var
