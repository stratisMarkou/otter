# %%
from pathlib import Path

import requests
import torchvision.transforms as transforms  # type: ignore
from matplotlib import pyplot as plt
from PIL import Image

from otter.experiment.downsample import downsample


def make_downsample_ladder_plot() -> plt.Figure:  # type: ignore
    url = "https://picsum.photos/256"  # Random image
    image = Image.open(requests.get(url, stream=True).raw)
    image2 = Image.open(requests.get(url, stream=True).raw)

    # Join 2 images together side-by-side, to get the lat > lon situation.
    joined_image = Image.new("RGB", (image.width * 2, image.height))
    joined_image.paste(image, (0, 0))
    joined_image.paste(image2, (image.width, 0))

    transform = transforms.ToTensor()
    tensor_image = transform(joined_image)  # 3xHxW

    height = tensor_image.shape[1]
    width = tensor_image.shape[2]

    # Make it into a 5D tensor, matching a batch of weather data.
    og = tensor_image.reshape(
        1, 3, height, width, 1
    )  # "Batch", C, H, W, "Time"
    og = og.permute(0, 2, 3, 4, 1)  # B, H, W, T, C

    image_tensors = []
    K = 10
    for k in range(0, K):
        lat = og.shape[1]
        lon = og.shape[2]
        n_channels = og.shape[4]

        scale = 2**k
        print(scale)
        downscaled = downsample(og, scale)
        result1 = downscaled[0, :, :, 0, :]
        plottable = result1.reshape(lat, lon, n_channels)
        image_tensors.append(plottable)

    fig, ax = plt.subplots(1, len(image_tensors), figsize=(15, 5))
    for k, im_tensor in enumerate(image_tensors):
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].imshow(im_tensor)
        ax[k].set_title(f"$k={k}$")

    return fig


if __name__ == "__main__":
    path = Path("_plots")
    path.mkdir(exist_ok=True)
    fig = make_downsample_ladder_plot()
    fig.savefig(path / "downsample_ladder.png", bbox_inches="tight")
