"""Stage-2 cross-attention cache state shared across FSDP2 kwargs copies."""

from __future__ import annotations


class Stage2CrossKVInitState:
    """Mutable leaf whose identity survives PyTorch's recursive tree rebuilds.

    FSDP2 recursively recreates ``dict`` and ``list`` arguments while moving
    and mixed-precision casting tensor inputs.  A plain boolean stored in a
    cache dictionary is therefore updated only in the innermost copy.  This
    deliberately non-dataclass leaf is passed through those rebuilds by
    identity, so attention and rollout audit observe the same state.
    """

    __slots__ = ("initialized",)

    def __init__(self, initialized: bool = False) -> None:
        if not isinstance(initialized, bool):
            raise TypeError("Stage-2 cross-KV initialized state must be a bool")
        self.initialized = initialized

    def mark_initialized(self) -> None:
        self.initialized = True

    def clear(self) -> None:
        self.initialized = False
