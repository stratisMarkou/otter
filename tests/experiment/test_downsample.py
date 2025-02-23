import pytest
import torch
import torch.testing

from otter.experiment.downsample import downsample


@pytest.fixture
def constant_tensor() -> torch.Tensor:
    """Create a simple RGB test tensor with known values."""
    # Create 8x8 RGB image as (1, 8, 8, 1, 3) tensor
    # Format: (Batch, Lat, Lon, Time, Var[RGB])
    r = torch.ones(8, 8) * 0.9
    g = torch.ones(8, 8) * 0.5
    b = torch.ones(8, 8) * 0.1
    rgb = torch.stack([r, g, b], dim=-1)  # 8x8x3
    rgb_reshaped = rgb.unsqueeze(0).unsqueeze(3)  # 1x8x8x1x3
    # Stack two copies into a "batch": 2x8x8x1x3
    rgb_batched = rgb_reshaped.repeat(2, 1, 1, 1, 1)
    return rgb_batched


@pytest.fixture
def gradient_tensor() -> torch.Tensor:
    """Create a simple gradient tensor for testing."""
    x = torch.linspace(0, 1, 16).repeat(16, 1)
    y = torch.linspace(0, 1, 16).unsqueeze(1).repeat(1, 16)
    pattern1 = (x + y) / 2
    pattern1 = pattern1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 1x16x16x1x1

    # Pattern 2 is reversed:
    pattern2 = 1 - pattern1
    return torch.cat([pattern1, pattern2], dim=0)  # 2x16x16x1x1


@pytest.fixture
def random_tensor() -> torch.Tensor:
    return torch.randn(4, 16, 16, 2, 4)


def test_basic_downsampling(constant_tensor: torch.Tensor) -> None:
    """Test basic downsampling with divisor=2 on simple RGB tensor."""
    result = downsample(constant_tensor, divisor=2)

    # Check shape preservation
    assert result.shape == constant_tensor.shape

    # Since we used constant values for each channel, downsampling
    # should preserve these values exactly
    torch.testing.assert_close(result[0, :, :, 0, 0], torch.ones(8, 8) * 0.9)
    torch.testing.assert_close(result[0, :, :, 0, 1], torch.ones(8, 8) * 0.5)
    torch.testing.assert_close(result[0, :, :, 0, 2], torch.ones(8, 8) * 0.1)


def test_transitivity(gradient_tensor: torch.Tensor) -> None:
    """Test that A->B->C equals A->C for spatial downsampling."""

    # Two-step downsampling: 16->8->4
    # NOTE: the second step needs to be 4, not 2, because the actual shape of
    # the tensor is unchanged after one downsampling step.
    two_step = downsample(downsample(gradient_tensor, 2), 4)

    # Direct downsampling: 16->4
    one_step = downsample(gradient_tensor, 4)

    torch.testing.assert_close(one_step, two_step)


def test_shape_preservation() -> None:
    """Test that shape is preserved for different input dimensions."""
    shapes = [
        (1, 16, 16, 1, 1),  # Single variable
        (2, 16, 16, 3, 4),  # Multiple batches, times, and variables
        (5, 15, 17, 2, 3),  # Odd spatial dimensions
    ]

    for shape in shapes:
        x = torch.randn(shape)
        for divisor in [2, 3, 4]:
            result = downsample(x, divisor)
            assert (
                result.shape == shape
            ), f"Shape mismatch for {shape} with divisor {divisor}"


def test_divisor_one(random_tensor: torch.Tensor) -> None:
    """Test that divisor=1 returns identical tensor."""
    result = downsample(random_tensor, 1)
    torch.testing.assert_close(result, random_tensor)


def test_extreme_downsampling(random_tensor: torch.Tensor) -> None:
    """Test downsampling to 1x1 spatial dimensions."""
    large_divisor = 100  # Forces 1x1 output
    result = downsample(random_tensor, large_divisor)

    # Check shape preservation
    assert result.shape == random_tensor.shape

    # Check that spatial dimensions are constant (1x1 repeated)
    for t in range(2):  # time
        for v in range(3):  # var
            unique_values = torch.unique(result[0, :, :, t, v])
            assert (
                len(unique_values) == 1
            ), "Spatial dimensions should be constant"


def test_mean_preservation(random_tensor: torch.Tensor) -> None:
    """Test that downsampling approximately preserves the mean."""
    for divisor in [2, 4, 8]:
        result = downsample(random_tensor, divisor)

        # Check means across spatial dimensions
        input_means = random_tensor.mean(dim=(1, 2))
        result_means = result.mean(dim=(1, 2))

        torch.testing.assert_close(input_means, result_means)


def test_batch_independence() -> None:
    """Test that batches are processed independently."""
    # Create two different batches
    batch1 = torch.ones(1, 8, 8, 2, 3)
    batch2 = torch.zeros(1, 8, 8, 2, 3)
    batched = torch.cat([batch1, batch2], dim=0)

    # Downsample separately and together
    result_separate = torch.cat(
        [downsample(batch1, 2), downsample(batch2, 2)], dim=0
    )
    result_batched = downsample(batched, 2)

    # Results should be identical
    torch.testing.assert_close(result_separate, result_batched)
