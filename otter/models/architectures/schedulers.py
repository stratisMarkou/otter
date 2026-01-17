from typing import Any, Dict

import numpy as np
import numpy.typing as npt


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
