from __future__ import annotations

import gc
from typing import Any

from .config import Settings, TensorCleanupPolicy


def memory_percent() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        return None


def release_tensors() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class TensorCleanupController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cleanup_count = 0

    def after_page(self, completed_pages: int) -> bool:
        policy = self.settings.tensor_cleanup_policy
        should_cleanup = False
        if policy is TensorCleanupPolicy.EVERY_N_PAGES:
            should_cleanup = completed_pages % self.settings.tensor_cleanup_every_pages == 0
        elif policy is TensorCleanupPolicy.ON_MEMORY_PRESSURE:
            percent = memory_percent()
            should_cleanup = percent is not None and percent >= self.settings.memory_pressure_percent
        if should_cleanup:
            release_tensors()
            self.cleanup_count += 1
        return should_cleanup

    def document_end(self) -> bool:
        if self.settings.tensor_cleanup_policy is TensorCleanupPolicy.DOCUMENT_END:
            release_tensors()
            self.cleanup_count += 1
            return True
        return False
