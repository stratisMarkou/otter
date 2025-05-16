import pytest
import torch

from otter.models.architectures.utils import pad_bottom_right

BATCH_SIZE = 4
IN_H = 121
IN_W = 240
N_CHANNELS = 80

OUT_H = 128
OUT_W = 256


@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, IN_H, IN_W, N_CHANNELS)


def test_pad(input_tensor: torch.Tensor) -> None:
    """Test that the pad function works correctly."""

    padded_tensor = pad_bottom_right(input_tensor, OUT_H, OUT_W)
    assert padded_tensor.shape == (
        BATCH_SIZE,
        OUT_H,
        OUT_W,
        N_CHANNELS,
    )
    assert torch.allclose(
        padded_tensor[:, :IN_H, :IN_W, :],
        input_tensor[:, :, :, :],
    )
    assert torch.all(padded_tensor[:, IN_H:, IN_W:, :] == 0)
