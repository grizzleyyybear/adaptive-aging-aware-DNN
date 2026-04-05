from __future__ import annotations

import os

import torch


def get_device_request(config=None, default: str = "auto") -> str:
    """Resolve the preferred device from config or environment."""
    request = None

    if config is not None and hasattr(config, "get"):
        runtime_cfg = config.get("runtime", {})
        if hasattr(runtime_cfg, "get"):
            request = runtime_cfg.get("device", None)
        if request is None:
            request = config.get("device", None)

    if request is None:
        request = os.environ.get("AGING_AWARE_DEVICE", default)

    return str(request).strip()


def resolve_device(request: str | None = None) -> torch.device:
    """Return a torch device, failing loudly if CUDA is explicitly requested but unavailable."""
    normalized = (request or os.environ.get("AGING_AWARE_DEVICE", "auto")).strip().lower()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see a GPU in this process. "
                "Check that the correct environment is active and that the process has GPU access."
            )
        return torch.device(normalized)

    return torch.device(normalized)


def configure_torch_runtime(device: torch.device) -> None:
    """Enable the common fast paths once CUDA is active."""
    if device.type != "cuda":
        return

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def dataloader_kwargs(device: torch.device) -> dict:
    """Pick sensible DataLoader settings for the active device."""
    use_cuda = device.type == "cuda"
    cpu_count = os.cpu_count() or 1
    num_workers = min(4, cpu_count) if use_cuda else 0

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def use_non_blocking(device: torch.device) -> bool:
    return device.type == "cuda"


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"

    try:
        return f"{device} ({torch.cuda.get_device_name(device)})"
    except Exception:
        return str(device)
