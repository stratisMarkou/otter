"""
Azure debugging utilities for diagnosing issues during initialization and checkpoint loading.

This module provides utilities to dump directory contents, file sizes, and other
diagnostic information when errors occur in Azure deployments.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def format_file_size(size_bytes: int | float) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def format_path_info(path: Path) -> str:
    """
    Return detailed info about a single path.

    Args:
        path: Path to inspect.

    Returns:
        String with path details (exists, type, size, permissions, symlink target).
    """
    if not path.exists():
        return f"  {path}: DOES NOT EXIST"

    try:
        stat = path.stat()
        size = format_file_size(stat.st_size)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        path_type = "dir" if path.is_dir() else "file"
        symlink_info = f" -> {os.readlink(path)}" if path.is_symlink() else ""

        return f"  {path.name}: {path_type}, {size}, modified {mtime}{symlink_info}"
    except Exception as e:
        return f"  {path}: ERROR reading info: {e}"


def format_directory_tree(
    path: Path,
    max_depth: int = 3,
    current_depth: int = 0,
    prefix: str = "",
) -> str:
    """
    Recursively list directory contents with file sizes and modification times.

    Args:
        path: Root path to list.
        max_depth: Maximum depth to recurse.
        current_depth: Current recursion depth (internal use).
        prefix: Line prefix for tree formatting (internal use).

    Returns:
        Formatted string with directory tree.
    """
    if not path.exists():
        return f"{prefix}{path}: DOES NOT EXIST\n"

    lines = []

    try:
        stat = path.stat()
        if path.is_file():
            size = format_file_size(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{prefix}{path.name} ({size}, {mtime})")
            return "\n".join(lines)

        # It's a directory
        if current_depth == 0:
            lines.append(f"{path}/")

        if current_depth >= max_depth:
            lines.append(f"{prefix}  ... (max depth reached)")
            return "\n".join(lines)

        try:
            children = sorted(path.iterdir())
        except PermissionError:
            lines.append(f"{prefix}  ... (permission denied)")
            return "\n".join(lines)

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                lines.append(
                    format_directory_tree(
                        child,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        prefix=child_prefix,
                    )
                )
            else:
                try:
                    stat = child.stat()
                    size = format_file_size(stat.st_size)
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%m-%d %H:%M"
                    )
                    lines.append(f"{prefix}{connector}{child.name} ({size}, {mtime})")
                except Exception as e:
                    lines.append(f"{prefix}{connector}{child.name} (error: {e})")

    except Exception as e:
        lines.append(f"{prefix}ERROR listing {path}: {e}")

    return "\n".join(line for line in lines if line)


def get_disk_usage_info(path: Path) -> str:
    """
    Get disk usage information for the filesystem containing the given path.

    Args:
        path: Path within the filesystem to check.

    Returns:
        Formatted string with disk usage information.
    """
    try:
        # Find an existing path to check
        check_path = path
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent

        if not check_path.exists():
            return "Disk info: Cannot determine (no accessible path)"

        usage = shutil.disk_usage(check_path)
        total = format_file_size(usage.total)
        used = format_file_size(usage.used)
        free = format_file_size(usage.free)
        percent_used = (usage.used / usage.total) * 100

        return f"Disk usage at {check_path}: {used} used / {total} total ({percent_used:.1f}% used), {free} free"
    except Exception as e:
        return f"Disk info: ERROR getting disk usage: {e}"


def log_checkpoint_diagnostics(
    checkpoint_dir: Path,
    error: Optional[Exception] = None,
) -> None:
    """
    Log comprehensive diagnostic info when checkpoint operations fail.

    Args:
        checkpoint_dir: The checkpoint directory to diagnose.
        error: The exception that was raised (if any).
    """
    lines = [
        "",
        "=" * 60,
        "CHECKPOINT DIAGNOSTICS",
        "=" * 60,
    ]

    if error is not None:
        lines.append(f"Error: {type(error).__name__}: {error}")

    lines.append(f"Checkpoint directory: {checkpoint_dir}")
    lines.append("")

    # Check for .tmp files that might indicate interrupted writes
    if checkpoint_dir.exists():
        tmp_files = list(checkpoint_dir.glob("*.tmp"))
        if tmp_files:
            lines.append("WARNING: Found .tmp files (possible interrupted writes):")
            for tmp in tmp_files:
                lines.append(format_path_info(tmp))
            lines.append("")

    # Directory contents
    lines.append("Checkpoint directory contents:")
    lines.append(format_directory_tree(checkpoint_dir, max_depth=2))
    lines.append("")

    # Parent directory contents
    if checkpoint_dir.parent.exists():
        lines.append(f"Parent directory ({checkpoint_dir.parent}):")
        lines.append(format_directory_tree(checkpoint_dir.parent, max_depth=1))
        lines.append("")

    # Disk usage
    lines.append(get_disk_usage_info(checkpoint_dir))

    lines.append("=" * 60)

    logging.error("\n".join(lines))


def log_experiment_diagnostics(
    experiment_path: Path,
    error: Optional[Exception] = None,
) -> None:
    """
    Log diagnostic info about an experiment directory.

    Called during experiment initialization to provide visibility into the
    directory state, useful for debugging Azure deployments.

    Args:
        experiment_path: The experiment directory to diagnose.
        error: Optional exception if an error occurred.
    """
    lines = [
        "",
        "=" * 60,
        "EXPERIMENT INITIALIZATION DIAGNOSTICS",
        "=" * 60,
    ]

    if error is not None:
        lines.append(f"Error: {type(error).__name__}: {error}")

    lines.append(f"Experiment path: {experiment_path}")
    lines.append("")

    # Check for config file
    config_path = experiment_path / "config.yaml"
    lines.append(f"Config file ({config_path}):")
    if config_path.exists():
        lines.append(format_path_info(config_path))
    else:
        lines.append("  DOES NOT EXIST")
    lines.append("")

    # Experiment directory contents
    lines.append("Experiment directory contents:")
    lines.append(format_directory_tree(experiment_path, max_depth=2))
    lines.append("")

    # Checkpoint directory status
    checkpoint_path = experiment_path / "checkpoints"
    lines.append(f"Checkpoint directory ({checkpoint_path}):")
    if checkpoint_path.exists():
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        lines.append(f"  Found {len(checkpoint_files)} checkpoint file(s)")
        for cp in sorted(checkpoint_files):
            lines.append(format_path_info(cp))
    else:
        lines.append("  DOES NOT EXIST")
    lines.append("")

    # Parent directory (results directory)
    if experiment_path.parent.exists():
        lines.append(f"Results directory ({experiment_path.parent}):")
        try:
            experiments = sorted(
                [d.name for d in experiment_path.parent.iterdir() if d.is_dir()],
                reverse=True,
            )
            lines.append(f"  Found {len(experiments)} experiment(s), showing last 20:")
            lines.append("")
            for exp in experiments[:20]:
                lines.append(f"    {exp}")
            if len(experiments) > 20:
                lines.append("")
                lines.append(f"  ... and {len(experiments) - 20} more")
        except Exception as e:
            lines.append(f"  ERROR listing: {e}")
    lines.append("")

    # Disk usage
    lines.append(get_disk_usage_info(experiment_path))

    lines.append("=" * 60)

    logging.error("\n".join(lines))
