"""
VRAM & System Memory Manager with Auto-OOM Recovery Handler
Provides memory monitoring, automatic sequential cache eviction, and graceful degradation.
"""

import gc
import os
import sys
import psutil
from typing import Dict, Any, Optional, Tuple, Callable
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("Avenox.VRAMManager")

# Lazy import torch
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    logger.warning("PyTorch belum terinstal di environment saat ini.")


def get_optimal_device() -> Tuple[str, str]:
    """
    Detects the best available hardware device.
    Returns: (device_type, device_name) e.g., ('cuda', 'NVIDIA GeForce RTX 3050')
    """
    if _torch_available and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", gpu_name
    
    # Try DirectML if torch_directml is available
    try:
        import torch_directml
        if torch_directml.is_available():
            return "dml", "DirectML Compatible GPU"
    except ImportError:
        pass

    return "cpu", f"CPU ({psutil.cpu_count(logical=True)} Threads)"


def get_vram_info() -> Dict[str, Any]:
    """
    Returns current GPU VRAM statistics (Total, Allocated, Reserved, Free) in Megabytes.
    """
    if not _torch_available or not torch.cuda.is_available():
        return {
            "available": False,
            "total_mb": 0,
            "allocated_mb": 0,
            "reserved_mb": 0,
            "free_mb": 0,
            "device_name": "CPU Only"
        }

    device_idx = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)
    allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
    free = total - reserved

    return {
        "available": True,
        "total_mb": round(total, 1),
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free, 1),
        "device_name": torch.cuda.get_device_name(device_idx)
    }


def get_ram_info() -> Dict[str, Any]:
    """
    Returns system RAM statistics in Megabytes.
    """
    ram = psutil.virtual_memory()
    return {
        "total_mb": round(ram.total / (1024 * 1024), 1),
        "used_mb": round(ram.used / (1024 * 1024), 1),
        "free_mb": round(ram.available / (1024 * 1024), 1),
        "percent_used": ram.percent
    }


def clear_gpu_cache(aggressive: bool = True) -> None:
    """
    Frees cached GPU memory buffers and forces Python garbage collection.
    Crucial for Low-VRAM 4GB sequential pipeline.
    """
    gc.collect()
    if _torch_available and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    logger.debug("GPU cache & garbage collection cleared.")


def unload_model(model: Any) -> None:
    """
    Safely unloads a PyTorch model from GPU memory to 0 MB residual VRAM.
    """
    if model is None:
        return
    
    try:
        if hasattr(model, "cpu"):
            model.cpu()
        del model
    except Exception as e:
        logger.warning(f"Error during model unload: {e}")
    finally:
        clear_gpu_cache(aggressive=True)


class AutoOOMRecovery:
    """
    Context manager / recovery handler that catches CUDA OOM exceptions,
    dynamically downscales chunk size, or gracefully falls back to CPU execution.
    """

    def __init__(
        self,
        current_chunk_size: float,
        min_chunk_size: float = 2.0,
        downscale_factor: float = 0.5,
        on_downscale_callback: Optional[Callable[[float, str], None]] = None
    ):
        self.chunk_size = current_chunk_size
        self.min_chunk_size = min_chunk_size
        self.downscale_factor = downscale_factor
        self.on_downscale_callback = on_downscale_callback
        self.fallback_to_cpu = False
        self.retry_needed = False

    def handle_oom(self, error: Exception) -> bool:
        """
        Handles an OutOfMemory error.
        Returns True if recovery strategy was applied and execution should be retried,
        or False if minimum limits were reached.
        """
        logger.warning(f"CUDA OOM terdeteksi! Memulai prosedur auto-recovery...")
        clear_gpu_cache(aggressive=True)

        new_chunk = self.chunk_size * self.downscale_factor
        if new_chunk >= self.min_chunk_size:
            self.chunk_size = round(new_chunk, 1)
            msg = f"Menurunkan ukuran chunk menjadi {self.chunk_size} detik untuk menghemat VRAM..."
            logger.info(msg)
            if self.on_downscale_callback:
                self.on_downscale_callback(self.chunk_size, msg)
            self.retry_needed = True
            return True
        elif not self.fallback_to_cpu:
            self.fallback_to_cpu = True
            msg = "Ukuran chunk sudah minimum. Beralih ke mode fallback CPU (FP16/FP32)..."
            logger.warning(msg)
            if self.on_downscale_callback:
                self.on_downscale_callback(self.chunk_size, msg)
            self.retry_needed = True
            return True
        else:
            logger.error("Auto-OOM Recovery gagal: Memori tidak mencukupi bahkan di CPU.")
            return False
