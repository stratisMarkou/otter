"""Tests for the Azure debugging utilities."""

from pathlib import Path

import pytest

from otter.experiment.azure_debug import (
    format_directory_tree,
    format_file_size,
    format_path_info,
    get_disk_usage_info,
    log_checkpoint_diagnostics,
    log_experiment_diagnostics,
)


# format_file_size tests


def test_format_file_size_bytes() -> None:
    assert format_file_size(100) == "100.0B"


def test_format_file_size_kilobytes() -> None:
    assert format_file_size(1024) == "1.0KB"
    assert format_file_size(2048) == "2.0KB"


def test_format_file_size_megabytes() -> None:
    assert format_file_size(1024 * 1024) == "1.0MB"


def test_format_file_size_gigabytes() -> None:
    assert format_file_size(1024 * 1024 * 1024) == "1.0GB"


# format_path_info tests


def test_format_path_info_nonexistent() -> None:
    result = format_path_info(Path("/nonexistent/path/to/file"))
    assert "DOES NOT EXIST" in result


def test_format_path_info_existing_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    result = format_path_info(test_file)
    assert "file" in result
    assert "12.0B" in result or "12" in result


def test_format_path_info_existing_directory(tmp_path: Path) -> None:
    result = format_path_info(tmp_path)
    assert "dir" in result


# format_directory_tree tests


def test_format_directory_tree_nonexistent() -> None:
    result = format_directory_tree(Path("/nonexistent/directory"))
    assert "DOES NOT EXIST" in result


def test_format_directory_tree_empty(tmp_path: Path) -> None:
    result = format_directory_tree(tmp_path)
    assert str(tmp_path.name) in result or str(tmp_path) in result


def test_format_directory_tree_with_files(tmp_path: Path) -> None:
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.txt").write_text("world")

    result = format_directory_tree(tmp_path)
    assert "file1.txt" in result
    assert "file2.txt" in result


def test_format_directory_tree_nested(tmp_path: Path) -> None:
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested content")

    result = format_directory_tree(tmp_path, max_depth=3)
    assert "subdir" in result
    assert "nested.txt" in result


def test_format_directory_tree_max_depth(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "d"
    nested.mkdir(parents=True)
    (nested / "deep.txt").write_text("deep")

    result = format_directory_tree(tmp_path, max_depth=1)
    assert "max depth reached" in result or "a" in result


# get_disk_usage_info tests


def test_get_disk_usage_info_existing_path(tmp_path: Path) -> None:
    result = get_disk_usage_info(tmp_path)
    assert "Disk usage" in result
    assert "used" in result
    assert "free" in result


def test_get_disk_usage_info_nonexistent_finds_parent() -> None:
    result = get_disk_usage_info(Path("/tmp/nonexistent/path"))
    assert "Disk usage" in result or "Cannot determine" in result


# log_checkpoint_diagnostics tests


def test_log_checkpoint_diagnostics_logs_contents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step_00001000.pt").write_bytes(b"x" * 1000)
    (checkpoint_dir / "latest.pt").write_bytes(b"y" * 500)

    log_checkpoint_diagnostics(checkpoint_dir)

    log_output = caplog.text
    assert "CHECKPOINT DIAGNOSTICS" in log_output
    assert "step_00001000.pt" in log_output
    assert "latest.pt" in log_output


def test_log_checkpoint_diagnostics_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    error = FileNotFoundError("No checkpoint found")
    log_checkpoint_diagnostics(tmp_path, error=error)

    log_output = caplog.text
    assert "FileNotFoundError" in log_output
    assert "No checkpoint found" in log_output


def test_log_checkpoint_diagnostics_logs_tmp_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pt.tmp").write_bytes(b"partial")

    log_checkpoint_diagnostics(checkpoint_dir)

    log_output = caplog.text
    assert ".tmp files" in log_output or "interrupted writes" in log_output


# log_experiment_diagnostics tests


def test_log_experiment_diagnostics_logs_contents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exp_path = tmp_path / "my_experiment"
    exp_path.mkdir()
    (exp_path / "config.yaml").write_text("model: test")

    log_experiment_diagnostics(exp_path)

    log_output = caplog.text
    assert "EXPERIMENT INITIALIZATION DIAGNOSTICS" in log_output
    assert "config.yaml" in log_output


def test_log_experiment_diagnostics_logs_missing_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exp_path = tmp_path / "my_experiment"
    exp_path.mkdir()

    log_experiment_diagnostics(exp_path)

    log_output = caplog.text
    assert "DOES NOT EXIST" in log_output


def test_log_experiment_diagnostics_logs_checkpoint_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exp_path = tmp_path / "my_experiment"
    exp_path.mkdir()
    (exp_path / "checkpoints").mkdir()
    (exp_path / "checkpoints" / "step_001.pt").write_bytes(b"ckpt")

    log_experiment_diagnostics(exp_path)

    log_output = caplog.text
    assert "1 checkpoint file" in log_output
