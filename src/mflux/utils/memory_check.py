"""Utility functions for memory management."""

import platform
import subprocess
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MemoryInfo:
    """System memory information."""

    total_memory: float  # Total physical memory in GB
    available_memory: float  # Available memory in GB
    is_apple_silicon: bool  # Whether running on Apple Silicon


@dataclass
class TrainingParams:
    """Safe training parameters based on available memory."""

    batch_size: int
    gradient_accumulation_steps: int
    max_train_steps: int


def get_memory_info() -> MemoryInfo:
    """Get system memory information."""
    is_apple_silicon = platform.processor() == "arm"

    try:
        # Get total physical memory
        total_mem_cmd = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        total_memory = int(total_mem_cmd.stdout.split()[1]) / (1024**3)  # Convert to GB

        # Get page size
        page_size_cmd = subprocess.run(["sysctl", "hw.pagesize"], capture_output=True, text=True)
        page_size = int(page_size_cmd.stdout.split()[1])

        # Get vm_stat output
        vm_stat_cmd = subprocess.run(["vm_stat"], capture_output=True, text=True)
        vm_stat_lines = vm_stat_cmd.stdout.split("\n")

        # Parse vm_stat output for free pages
        free_pages = 0
        for line in vm_stat_lines:
            if "Pages free:" in line:
                free_pages = int(line.strip().split()[2].rstrip("."))
                break

        # Calculate available memory in GB
        available_memory = (free_pages * page_size) / (1024**3)

        return MemoryInfo(
            total_memory=total_memory, available_memory=available_memory, is_apple_silicon=is_apple_silicon
        )
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"Warning: Using conservative memory estimates due to: {e}")
        # Fallback to conservative estimates for M1/M2 Macs
        if is_apple_silicon:
            return MemoryInfo(
                total_memory=16.0,  # Conservative estimate
                available_memory=8.0,
                is_apple_silicon=True,
            )
        else:
            return MemoryInfo(total_memory=8.0, available_memory=4.0, is_apple_silicon=False)


def calculate_safe_training_params(
    memory_info: MemoryInfo,
    requested_batch_size: int = 1,
    requested_grad_accum: int = 1,
    requested_steps: int = 200,
) -> Tuple[TrainingParams, str]:
    """Calculate safe training parameters based on available memory.

    Args:
        memory_info: System memory information
        requested_batch_size: Requested batch size
        requested_grad_accum: Requested gradient accumulation steps
        requested_steps: Requested number of training steps

    Returns:
        Tuple of (safe training parameters, warning message if parameters were adjusted)
    """
    # Estimate memory requirements (in GB)
    # These are rough estimates and may need adjustment based on actual usage
    base_memory = 4.0  # Base memory for model and other overhead
    per_sample_memory = 0.5  # Memory per sample in batch

    total_batch_memory = base_memory + (per_sample_memory * requested_batch_size * requested_grad_accum)

    # Leave some headroom for system
    safe_memory_limit = memory_info.available_memory * 0.8

    if total_batch_memory > safe_memory_limit:
        # Calculate safe batch size
        safe_batch = max(1, int((safe_memory_limit - base_memory) / per_sample_memory))

        # Adjust steps to maintain similar total training samples
        original_samples = requested_batch_size * requested_grad_accum * requested_steps
        safe_steps = min(1000, int(original_samples / safe_batch))  # Cap at 1000 steps

        warning = (
            f"Memory requirements ({total_batch_memory:.1f}GB) might exceed safe limit "
            f"({safe_memory_limit:.1f}GB). Parameters have been adjusted for stability."
        )

        return TrainingParams(batch_size=1, gradient_accumulation_steps=safe_batch, max_train_steps=safe_steps), warning

    return TrainingParams(
        batch_size=requested_batch_size,
        gradient_accumulation_steps=requested_grad_accum,
        max_train_steps=requested_steps,
    ), ""
