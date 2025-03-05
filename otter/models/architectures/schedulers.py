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

    def _sample_data_scale(self) -> int:
        # Sample according to scale probability
        return int(
            self.data_scales[torch.multinomial(self.scale_probs, 1)].item()
        )

    def get_scales(self) -> torch.Tensor:
        """
        Returns the entire sequence of scales through which the target data
        will be processed.

        Here, the first scale is the scale by which the true target data is
        downsampled, and the subsequent scales are the scales at which the
        model will roll out predictions.

        For example, during training, this function might return [32],
        indicating that the true target data is downsampled by a factor of 32,
        and the model will not roll out predictions.

        During validation, we might return [128, 64, 32, 16, 8, 4, 2],
        indicating that the true target data is downsampled by a factor of 128,
        (i.e. erased), and the model will roll out predictions at scales 64,
        32, 16, 8, 4, and 2.
        """
        true_trg_scale = self._sample_data_scale()
        merged_scales = [true_trg_scale] + self.rollout_scales
        return torch.tensor(merged_scales, dtype=torch.int32)
