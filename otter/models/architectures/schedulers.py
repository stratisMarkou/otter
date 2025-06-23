import numpy as np
import numpy.typing as npt
import torch
from typing import Any, Dict


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


class ForecastHorizonScheduler:
    def __init__(
        self,
        num_steps: int,
        max_frame_forecast_horizon: int,
        min_frame_forecast_horizon: int,
        temporal_resolution_hours: int,
    ) -> None:
        """
        Args:
            num_steps: total number of steps in the training process.
            max_frame_forecast_horizon: maximum forecast horizon in frames.
            min_frame_forecast_horizon: minimum forecast horizon in frames.
            temporal_resolution_hours: temporal resolution of the data in hours.
        """
        super().__init__()

        assert max_frame_forecast_horizon >= min_frame_forecast_horizon, (
            f"Max forecast horizon {max_frame_forecast_horizon} "
            f"must be greater than or equal to min forecast horizon "
            f"{min_frame_forecast_horizon}"
        )

        self.num_steps = num_steps
        self.max_frame_forecast_horizon = max_frame_forecast_horizon
        self.min_frame_forecast_horizon = min_frame_forecast_horizon

        self.current_frame_forecast_horizon = min_frame_forecast_horizon
        self.current_step = 0

        self.increment_every_num_steps = num_steps // (
            max_frame_forecast_horizon - min_frame_forecast_horizon + 1
        )

        self.temporal_resolution_hours = temporal_resolution_hours

    def step(self) -> None:
        """
        Increments the current step and updates the forecast horizon if is devisible by `increment_every_num_steps`.
        """
        self.current_step += 1

        if self.current_step % self.increment_every_num_steps == 0:
            if (
                self.current_frame_forecast_horizon
                < self.max_frame_forecast_horizon
            ):
                self.current_frame_forecast_horizon += 1

    def get_frame_forecast_horizon(self) -> int:
        """
        Returns the current forecast horizon in frames.
        """
        return self.current_frame_forecast_horizon

    def get_temporal_forecast_horizon_hours(self) -> npt.NDArray[np.int32]:
        """
        Returns the current forecast horizon in hours.
        """
        return self.temporal_resolution_hours * (
            np.arange(self.current_frame_forecast_horizon) + 1
        )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the scheduler's state as a dictionary."""
        return dict(self.__dict__)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
