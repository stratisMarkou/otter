"""Checks that determinstic config initializes as expected."""

from otter.experiment.utils import initialize_experiment


def test_deterministic_config() -> None:

    initialize_experiment(
        ["--debug", "--config", "otter/experiment/config/deterministic.yaml"]
    )
