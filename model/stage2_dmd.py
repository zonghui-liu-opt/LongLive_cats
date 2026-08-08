"""Stage-2 model-role container.

This module deliberately contains no score adapter, rollout, loss, optimizer,
EMA, data loader, T5, or VAE construction.  Batch 2 only establishes three
independent DiT roles; algorithmic forwards are added in later gated batches.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


class Stage2DiTRole(nn.Module):
    """A thin role wrapper around one independently-owned Wan DiT."""

    _ROLES = {"generator", "real_score", "fake_score"}

    def __init__(self, model: nn.Module, *, role: str, is_causal: bool):
        super().__init__()
        if role not in self._ROLES:
            raise ValueError(f"unknown Stage-2 role: {role!r}")
        if role == "generator" and not is_causal:
            raise ValueError("Stage-2 generator must be causal")
        if role != "generator" and is_causal:
            raise ValueError(f"Stage-2 {role} must be bidirectional")
        if not isinstance(model, nn.Module):
            raise TypeError("Stage2DiTRole.model must be torch.nn.Module")
        self.model = model
        self.role = role
        self.is_causal = bool(is_causal)
        self.uniform_timestep = not self.is_causal

    def forward(self, *args, **kwargs):  # pragma: no cover - safety tripwire
        raise RuntimeError(
            "Stage-2 Batch 2 is init-only; model forward is implemented in "
            "the later score/rollout batches."
        )


def _parameter_storage_ids(module: nn.Module) -> set[tuple[str, int]]:
    storages: set[tuple[str, int]] = set()
    for parameter in module.parameters():
        if parameter.is_meta or parameter.numel() == 0:
            continue
        local = parameter
        to_local = getattr(parameter, "to_local", None)
        if callable(to_local):
            local = to_local()
        if local.numel() == 0:
            continue
        storage = local.untyped_storage()
        storages.add((str(local.device), int(storage.data_ptr())))
    return storages


class Stage2DMD(nn.Module):
    """Role-only Stage-2 skeleton; no training or inference forward yet."""

    def __init__(
        self,
        *,
        generator: Stage2DiTRole,
        real_score: Stage2DiTRole,
        fake_score: Stage2DiTRole,
    ):
        super().__init__()
        expected = {
            "generator": generator,
            "real_score": real_score,
            "fake_score": fake_score,
        }
        for role, value in expected.items():
            if not isinstance(value, Stage2DiTRole) or value.role != role:
                raise ValueError(f"Stage-2 {role} wrapper has the wrong role")
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.audit_independent_role_storage()

    def _roles(self) -> Iterable[tuple[str, Stage2DiTRole]]:
        return (
            ("generator", self.generator),
            ("real_score", self.real_score),
            ("fake_score", self.fake_score),
        )

    def audit_independent_role_storage(self) -> dict[str, object]:
        roles = dict(self._roles())
        if len({id(value) for value in roles.values()}) != 3:
            raise RuntimeError(
                "Stage-2 role wrappers must be three independent objects"
            )
        if len({id(value.model) for value in roles.values()}) != 3:
            raise RuntimeError("Stage-2 role DiTs must be three independent objects")

        parameter_ids = {
            role: {id(parameter) for parameter in wrapper.parameters()}
            for role, wrapper in roles.items()
        }
        storage_ids = {
            role: _parameter_storage_ids(wrapper) for role, wrapper in roles.items()
        }
        role_names = tuple(roles)
        for left_index, left in enumerate(role_names):
            for right in role_names[left_index + 1 :]:
                if parameter_ids[left] & parameter_ids[right]:
                    raise RuntimeError(
                        f"Stage-2 roles share Parameter objects: {left}/{right}"
                    )
                if storage_ids[left] & storage_ids[right]:
                    raise RuntimeError(
                        f"Stage-2 roles share parameter storage: {left}/{right}"
                    )
        return {
            "parameter_objects_disjoint": True,
            "parameter_storage_disjoint": True,
            "roles": role_names,
        }

    def forward(self, *args, **kwargs):  # pragma: no cover - safety tripwire
        raise RuntimeError(
            "Stage-2 Batch 2 does not implement a DMD/DFD forward; use only "
            "the dedicated init-only preflight."
        )
