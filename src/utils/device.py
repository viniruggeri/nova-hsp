"""
Device detection and management utilities.

Automatically detects CUDA GPU availability and provides
device configuration for PyTorch models.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Global device cache
_device = None
_device_info = None


def get_device(
    force_cpu: bool = False, device_id: Optional[int] = None
) -> torch.device:
    """
    Get PyTorch device (CUDA GPU or CPU) with automatic detection.

    Args:
        force_cpu: If True, always use CPU even if CUDA is available
        device_id: Specific GPU device ID to use (default: 0)

    Returns:
        torch.device: PyTorch device object

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device(force_cpu=True)  # Force CPU
        >>> device = get_device(device_id=1)  # Use GPU 1
    """
    global _device

    # Check environment variable override
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        force_cpu = True

    # Return cached device if available and no override
    if _device is not None and not force_cpu:
        return _device

    # Check CUDA availability
    if force_cpu:
        device = torch.device("cpu")
        logger.info("Using CPU (forced)")
    elif torch.cuda.is_available():
        # Use specified GPU or default to 0
        gpu_id = device_id if device_id is not None else 0

        # Validate GPU ID
        if gpu_id >= torch.cuda.device_count():
            logger.warning(
                f"GPU {gpu_id} not available. "
                f"Available GPUs: {torch.cuda.device_count()}. Using GPU 0."
            )
            gpu_id = 0

        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using CUDA GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU")

    # Cache device
    _device = device
    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed information about the current device.

    Args:
        device: PyTorch device (if None, uses auto-detected device)

    Returns:
        dict: Device information including type, name, memory, etc.
    """
    global _device_info

    if device is None:
        device = get_device()

    # Return cached info if available
    if _device_info is not None and device == _device:
        return _device_info

    info = {
        "device": str(device),
        "type": device.type,
        "is_cuda": device.type == "cuda",
    }

    if device.type == "cuda":
        gpu_id = device.index if device.index is not None else 0
        info.update(
            {
                "gpu_id": gpu_id,
                "gpu_name": torch.cuda.get_device_name(gpu_id),
                "cuda_version": torch.version.cuda,
                "total_memory_gb": torch.cuda.get_device_properties(gpu_id).total_memory
                / 1e9,
                "available_gpus": torch.cuda.device_count(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    else:
        info.update(
            {
                "cpu_count": os.cpu_count(),
                "num_threads": torch.get_num_threads(),
            }
        )

    _device_info = info
    return info


def set_device(device: str) -> torch.device:
    """
    Manually set the device.

    Args:
        device: Device string ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        torch.device: PyTorch device object
    """
    global _device, _device_info

    _device = torch.device(device)
    _device_info = None  # Reset cache

    logger.info(f"Device manually set to: {_device}")
    return _device


def clear_device_cache():
    """Clear device cache (useful for testing or reinitializing)."""
    global _device, _device_info
    _device = None
    _device_info = None


def move_to_device(obj, device: Optional[torch.device] = None):
    """
    Move tensor or model to device.

    Args:
        obj: Tensor, model, or iterable of tensors/models
        device: Target device (if None, uses auto-detected device)

    Returns:
        Object moved to device
    """
    if device is None:
        device = get_device()

    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj


def print_device_info(device: Optional[torch.device] = None):
    """
    Print detailed device information.

    Args:
        device: PyTorch device (if None, uses auto-detected device)
    """
    info = get_device_info(device)

    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)

    for key, value in info.items():
        key_formatted = key.replace("_", " ").title()
        print(f"  {key_formatted:.<40} {value}")

    print("=" * 60 + "\n")


def set_reproducible(seed: int = 42):
    """
    Set random seeds for reproducibility across CPU and GPU.

    Args:
        seed: Random seed value
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed} for reproducibility")


# Initialize device on import
try:
    _default_device = get_device()
    logger.debug(f"Default device initialized: {_default_device}")
except Exception as e:
    logger.warning(f"Failed to initialize default device: {e}")
    _default_device = torch.device("cpu")
