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

        self.data_scales = torch.tensor(data_scales, dtype=torch.int32)
        self.rollout_scales = rollout_scales
        self.scale_probs = torch.tensor(data_scale_probs, dtype=torch.float32)

    def sample_data_scale(self) -> int:
        """
        Samples a scale for data preprocessing according to the distribution
        defined via `data_scales` and `data_scale_probs`.
        """
        # Sample according to scale probability
        return int(
            self.data_scales[torch.multinomial(self.scale_probs, 1)].item()
        )

    def get_rollout_scales(self) -> torch.Tensor:
        """
        Returns the scales through which the model's predictions will be
        rolled out.
        """
        return torch.tensor(self.rollout_scales, dtype=torch.int32)
