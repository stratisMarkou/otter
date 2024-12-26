"""Tests utilitiles used in swin transformer."""

import pytest
import torch

from otter.models.architectures.transformer_blocks import (
    combine_windows,
    extract_windows,
    shift_horizontally_and_vertically,
)


# Tests that combine_windows the is inverse of extract_windows.
@pytest.mark.parametrize("window_size", [2, 4])
def test_extract_then_combine_windows(window_size: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    x = torch.rand((B, H, W, D))
    extracted = extract_windows(x=x, window_size=window_size)
    y = combine_windows(
        x=extracted,
        original_shape=x.shape,
        window_size=window_size,
    )
    assert torch.allclose(x, y)


# Tests that extract_windows is the inverse of combine_windows.
@pytest.mark.parametrize("window_size", [2, 4])
def test_combine_then_extract_windows(window_size: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    S = window_size
    B = B * (H // S) * (W // S)

    x = torch.rand((B * (H // S) * (W // S), S**2, D))
    combined = combine_windows(
        x=x,
        original_shape=(B, H, W, D),
        window_size=window_size,
    )
    y = extract_windows(x=combined, window_size=window_size)
    assert torch.allclose(x, y)


# Tests that shifting horizontally and vertically is invertible.
@pytest.mark.parametrize("shift", [1, 2, 3, 4])
def test_shift_horizontally_and_vertically(shift: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    x = torch.rand((B, H, W, D))
    shifted = shift_horizontally_and_vertically(x=x, shift=shift)
    y = shift_horizontally_and_vertically(x=shifted, shift=-shift)
    assert torch.allclose(x, y)
