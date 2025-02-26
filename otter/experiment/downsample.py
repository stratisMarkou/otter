import torch
import torch.nn.functional as F


def downsample(original: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Downsample a batch of ERA5 data by a factor of `scale`.
    The input tensor is expected to have the shape (B, Lat, Lon, Time, Var),
    where
        B is the batch size,
        Lat and Lon are the spatial dimensions,
        Time is the time dimension,
        Var is the variable/channel dimension.

    NOTE: The output tensor will have the same shape as the input tensor, but
        "pixelated" by the downsampling factor.

    NOTE: If scale > min(num_lat, num_lon), the output tensor will not be the
        mean, but will instead be a tensor of zeros. This corresponds to the
        inference case, where we don't know the average weather data at the
        target time.

    Args:
        original: The input tensor.
        scale: The downsampling factor by which the spatial dimensions will be
            divided.
    """
    if scale == 1:
        # No downsampling needed
        return original

    batch, lat, lon, time, var = original.shape

    lat_down = lat // scale
    lon_down = lon // scale

    if lat_down < 1 or lon_down < 1:
        # This corresponds to the inference case, where we don't know the
        # weather data at the target time, (and also don't know the spatial
        # average). We return a tensor of zeros as a special case.
        return torch.zeros_like(original)

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
