"""Crash-safe checkpoint and file-write helpers.

These utilities make long PARAM/A100 runs failproof and resumable:

* ``atomic_torch_save`` / ``atomic_write_text`` write to a temporary file in
  the same directory and ``os.replace`` it into place. ``os.replace`` is
  atomic on POSIX and Windows, so a crash (or SLURM walltime kill) during a
  write can never leave a half-written, corrupt checkpoint or results file.
* ``safe_torch_load`` tolerates a missing or corrupt file and returns
  ``None`` instead of raising, so a damaged resume file degrades to "start
  fresh" rather than crashing the whole job.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    # Best-effort flush to disk before the atomic rename.
    try:
        with open(tmp_path, "rb") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass
    os.replace(tmp_path, final_path)


def atomic_torch_save(obj: Any, path: os.PathLike | str) -> Path:
    """Serialize ``obj`` to ``path`` atomically (tmp file + os.replace)."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".", suffix=".tmp", dir=str(final_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        _atomic_replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return final_path


def atomic_write_text(text: str, path: os.PathLike | str, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace)."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".", suffix=".tmp", dir=str(final_path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return final_path


def safe_torch_load(path: os.PathLike | str, map_location: Any = None) -> Optional[Any]:
    """Load a checkpoint, returning ``None`` if it is missing or corrupt."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except Exception:
        return None
