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
        num_steps: list[int],
        initial_learning_rates: list[float],
        warmup_steps: list[int],
        max_frame_forecast_horizon: int,
        min_frame_forecast_horizon: int,
        temporal_resolution_hours: int,
    ) -> None:
        """
        Args:
            num_steps: number of steps for each horizon
            initial_learning_rates: initial learning rates for each horizon
            warmup_steps: number of warmup steps for each horizon for lr scheduler
            max_frame_forecast_horizon: maximum forecast horizon in frames
            min_frame_forecast_horizon: minimum forecast horizon in frames
            temporal_resolution_hours: temporal resolution in hours
        """
        assert max_frame_forecast_horizon >= min_frame_forecast_horizon, (
            f"Max forecast horizon {max_frame_forecast_horizon} "
            f"must be greater than or equal to min forecast horizon "
            f"{min_frame_forecast_horizon}"
        )

        self.num_steps = num_steps
        self.max_frame_forecast_horizon = max_frame_forecast_horizon
        self.min_frame_forecast_horizon = min_frame_forecast_horizon

        assert (
            len(num_steps) == len(initial_learning_rates) == len(warmup_steps)
        ), (
            f"Mismatch between number of steps {len(num_steps)}, "
            f"initial learning rates {len(initial_learning_rates)}, "
            f"and warmup steps {len(warmup_steps)}"
        )

        assert (
            len(num_steps)
            == max_frame_forecast_horizon - min_frame_forecast_horizon + 1
        ), (
            f"Number of steps {len(num_steps)} must match the number of "
            f"forecast horizons {max_frame_forecast_horizon - min_frame_forecast_horizon + 1}"
        )

        self.initial_learning_rates = initial_learning_rates
        self.warmup_steps = warmup_steps

        self.current_frame_forecast_horizon = min_frame_forecast_horizon
        self.current_step = 0

        self.current_iteration_per_step = 0

        self.temporal_resolution_hours = temporal_resolution_hours

    def step(self) -> bool:
        """
        Increments the current step and updates the forecast horizon if is divisible by `increment_every_num_steps`.
        Return bool indicating whether the forecast horizon has been updated.
        """
        self.current_iteration_per_step += 1

        if (
            self.current_iteration_per_step
            >= self.num_steps[self.current_step]
        ):
            self.current_frame_forecast_horizon += 1

            self.current_step += 1
            self.current_iteration_per_step = 0

            return True

        return False

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

    def get_current_optimizer_params(self) -> Dict[str, Any]:
        """
        Returns the current optimizer parameters for the learning rate scheduler.
        """
        return {
            "lr": self.initial_learning_rates[self.current_step],
        }

    def get_current_lr_scheduler_params(self) -> Dict[str, Any]:
        """
        Returns the current learning rate scheduler parameters.
        """
        return {
            "num_steps": self.num_steps[self.current_step],
            "warmup_steps": self.warmup_steps[self.current_step],
        }
