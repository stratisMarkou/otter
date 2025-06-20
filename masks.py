# %%
import torch
from otter.plots.geoplot import GeoPlotter

path = "/home/jonas/Documents/code/otter/archesweather_constant_masks.pt"
masks = torch.load(path)
# %%
masks.shape


def reverse_default_preprocessor(tensor):
    # first
    half_lon = tensor.shape[-1] // 2
    tensor = torch.cat(
        [tensor[..., :, half_lon:], tensor[..., :, :half_lon]], dim=-1
    )
    # add back south pole
    tensor = torch.cat([tensor, tensor[..., -1:, :]], dim=-2)

    # flip back
    tensor = torch.flip(tensor, dims=[-2])

    return tensor


masks2 = reverse_default_preprocessor(masks)
# %%
gp = GeoPlotter()
gp.plot_single(masks2[3, 0, :, :])
# %%
masks2.shape
# %%
masks.std()
