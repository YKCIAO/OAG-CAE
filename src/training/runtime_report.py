from __future__ import annotations

import os
import platform
import time
from typing import Dict, Any

import torch


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_system_info(device) -> Dict[str, Any]:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": str(device),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "num_gpus": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
        })

    return info


def reset_peak_memory(device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device) -> float:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


class Timer:
    def __init__(self, device=None):
        self.device = device
        self.start_time = None

    def start(self):
        if self.device is not None and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.device is not None and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - self.start_time