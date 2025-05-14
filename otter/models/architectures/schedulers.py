import torch


class ScaleScheduler:
    def __init__(
        self,
        data_scales: list[int],
        data_scale_probs: list[float],
        rollout_scales: list[int],
    ) -> None:
        """
        Args:
            data_scales: list of scales to choose from for data preprocessing,
                applied to real target data.
            data_scale_probs: list of probabilities for sampling data_scales,
                must be of the same length as data_scales
            rollout_scales: deterministic hierarchy of scales to rollout for
                getting predictions
        """
        super().__init__()

        assert len(data_scales) == len(data_scale_probs), (
            f"Mismatch between number of data scales {len(data_scales)} "
            f"and number of associated probabilities {len(data_scale_probs)}"
        )
        assert data_scales == sorted(data_scales, reverse=True), (
            f"Data scales must be sorted descendingly, got {data_scales}"
        )
        assert rollout_scales == sorted(rollout_scales, reverse=True), (
            f"Rollout scales must be sorted descendingly, got {rollout_scales}"
        )

        self.data_scales = torch.tensor(data_scales, dtype=torch.int32)
        self.rollout_scales = rollout_scales
        self.scale_probs = torch.tensor(data_scale_probs, dtype=torch.float32)

    def sample_data_scale_pair(self) -> tuple[int, int]:
        """
        Samples a pair of scales:
         1. A scale for the downsampled target INPUT into the model
         2. A scale for the downsampled target that the model attempts to
            predict
        """
        idx = torch.multinomial(self.scale_probs, 1)
        input_scale = int(self.data_scales[idx].item())

        if idx == len(self.data_scales) - 1:
            # The last scale should target the clean image
            output_scale = 1
        else:
            output_scale = int(self.data_scales[idx + 1].item())
        return input_scale, output_scale

    def get_rollout_scales(self) -> torch.Tensor:
        """
        Returns the scales through which the model's predictions will be
        rolled out.
        """
        return torch.tensor(self.rollout_scales, dtype=torch.int32)
