import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(
        self,
        drop_prob: float,
        batch_size: int,
        scale_by_keep: bool = True,
    ):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        # Check that the input dimension is divisible by the batch size
        assert x.shape[0] % self.batch_size == 0, (
            f"Input tensor size ({x.shape[0]}) is not divisible by "
            f"batch_size ({self.batch_size})"
        )

        # Calculate the number of sequences folded into the batch
        num_sequences = x.shape[0] // self.batch_size

        # Create a random tensor for the *true* batch size
        # Shape: (self.batch_size,) -> e.g., (4,)
        base_random_tensor = x.new_empty((self.batch_size,)).bernoulli_(
            keep_prob
        )

        # Expand the random tensor to match the folded batch dimension
        expanded_random_tensor = base_random_tensor.repeat_interleave(
            num_sequences, dim=0
        )

        shape = (-1,) + (1,) * (x.ndim - 1)
        random_tensor = expanded_random_tensor.view(shape)

        # Scale by keep_prob
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        # Apply dropout
        return x * random_tensor
