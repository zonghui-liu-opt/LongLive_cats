"""A genuinely lazy W&B proxy used only by legacy logging paths."""

from __future__ import annotations

import importlib


class _LazyWandb:
    _module = None

    def _load(self):
        if self._module is None:
            try:
                self._module = importlib.import_module("wandb")
            except ImportError as exc:
                raise RuntimeError(
                    "W&B logging was enabled but the optional 'wandb' package is "
                    "not installed. Disable W&B or install the dependency."
                ) from exc
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


wandb = _LazyWandb()
