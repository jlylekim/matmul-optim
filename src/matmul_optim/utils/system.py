from __future__ import annotations

from typing import Any

import torch


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "cuda_runtime": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            }
        )
    return info
