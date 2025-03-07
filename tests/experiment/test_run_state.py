import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from otter.experiment.run_state import (
    RunState,
    load_initial_state,
    save_run_state,
)


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        assert isinstance(out, torch.Tensor)
        return out


@pytest.fixture
def model() -> SimpleModel:
    return SimpleModel()


@pytest.fixture
def optimizer(model: SimpleModel) -> Adam:
    return Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer: Adam) -> StepLR:
    return StepLR(optimizer, step_size=1)


@pytest.fixture
def scaler() -> GradScaler:
    return GradScaler("cpu")


@pytest.fixture
def run_state(
    model: SimpleModel, optimizer: Adam, scheduler: StepLR, scaler: GradScaler
) -> RunState:
    # Get initial states
    return RunState(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),  # type: ignore
        scaler_state=scaler.state_dict(),
        epoch=5,
        step=100,
    )


def assert_state_equal(state1: RunState, state2: RunState) -> None:
    """Compare two RunState objects for equality."""
    assert state1.epoch == state2.epoch, "Epochs do not match"
    assert state1.step == state2.step, "Steps do not match"

    # Compare model states
    for key in state1.model_state:
        assert torch.equal(state1.model_state[key], state2.model_state[key]), (
            f"Model states differ at key {key}"
        )

    # Compare optimizer states
    assert state1.optimizer_state.keys() == state2.optimizer_state.keys(), (
        "Optimizer state keys differ"
    )

    # Compare scheduler states
    assert state1.scheduler_state == state2.scheduler_state, (
        "Scheduler states differ"
    )

    # Compare scaler states
    assert state1.scaler_state == state2.scaler_state, "Scaler states differ"


def test_runstate_creation(run_state: RunState) -> None:
    """Test basic RunState creation and properties."""
    assert run_state.epoch == 5
    assert run_state.step == 100
    assert isinstance(run_state.model_state, dict)
    assert isinstance(run_state.optimizer_state, dict)
    assert isinstance(run_state.scheduler_state, dict)
    assert isinstance(run_state.scaler_state, dict)


def test_to_dict_from_dict(run_state: RunState) -> None:
    """Test conversion to and from dictionary."""
    state_dict: dict[str, Any] = run_state.to_dict()
    new_state: RunState = RunState.from_dict(state_dict)
    assert_state_equal(run_state, new_state)


def test_save_load(run_state: RunState) -> None:
    """Test saving and loading from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path: Path = Path(tmpdir) / "state.pt"

        run_state.save(path)
        assert path.exists(), "State file was not created"

        loaded_state: RunState = RunState.load(path)

        assert_state_equal(run_state, loaded_state)


def test_restore(
    run_state: RunState,
    model: SimpleModel,
    optimizer: Adam,
    scheduler: StepLR,
    scaler: GradScaler,
) -> None:
    """Test restoring state to PyTorch objects."""
    # Modify the objects to ensure they're different from initial state
    for param in model.parameters():
        param.data.fill_(1.0)
    optimizer.param_groups[0]["lr"] = 0.1

    optimizer.step()
    scheduler.step()

    run_state.restore(model, optimizer, scheduler, scaler)

    current_state: RunState = RunState(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),  # type: ignore
        scaler_state=scaler.state_dict(),
        epoch=run_state.epoch,
        step=run_state.step,
    )

    assert_state_equal(run_state, current_state)


def test_runstate_with_training(
    model: SimpleModel, optimizer: Adam, scheduler: StepLR, scaler: GradScaler
) -> None:
    """Test RunState in a simulated training scenario."""
    initial_state: RunState = RunState(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),  # type: ignore
        scaler_state=scaler.state_dict(),
        epoch=0,
        step=0,
    )

    # Simulate some training
    x: torch.Tensor = torch.randn(32, 10)
    y: torch.Tensor = torch.randint(0, 2, (32,))
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    with torch.autocast("cpu"):
        output: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(output, y)

    scaler.scale(loss).backward()  # type: ignore
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    trained_state: RunState = RunState(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),  # type: ignore
        scaler_state=scaler.state_dict(),
        epoch=1,
        step=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path: Path = Path(tmpdir) / "state.pt"
        trained_state.save(path)
        loaded_state: RunState = RunState.load(path)

        assert_state_equal(trained_state, loaded_state)

        # Verify it's different from initial state
        with pytest.raises(AssertionError):
            assert_state_equal(initial_state, loaded_state)


@dataclass
class MockExperimentInfo:
    experiment_path: str


@dataclass
class MockExperiment:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: GradScaler
    info: MockExperimentInfo


def test_get_latest_checkpoint(
    model: SimpleModel,
    optimizer: Adam,
    scheduler: StepLR,
    scaler: GradScaler,
) -> None:
    """Test the wrapping functions' ability to get the latest checkpoint."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path: Path = Path(tmpdir) / "experiment"
        experiment = MockExperiment(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            scaler=scaler,
            info=MockExperimentInfo(
                experiment_path=str(path),
            ),
        )

        # First state
        save_run_state(
            experiment, optimizer, scheduler, scaler, epoch=0, step=0
        )

        # Second state
        save_run_state(
            experiment, optimizer, scheduler, scaler, epoch=1, step=100
        )

        # Third state
        save_run_state(
            experiment, optimizer, scheduler, scaler, epoch=1, step=300
        )

        # Ensure all checkpoints are saved:
        assert (path / "checkpoints" / "step_0.pt").exists()
        assert (path / "checkpoints" / "step_100.pt").exists()
        assert (path / "checkpoints" / "step_300.pt").exists()

        # This method should find and load the latest checkpoint
        state = load_initial_state(experiment, optimizer, scheduler, scaler)
        assert state.epoch == 1
        assert state.step == 300
