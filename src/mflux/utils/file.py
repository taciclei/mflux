"""File utility functions."""

import os


def make_dir(path: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        path: Path to directory to create
    """
    os.makedirs(path, exist_ok=True)
