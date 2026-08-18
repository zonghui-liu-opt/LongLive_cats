"""Fail-closed parameter-name mapping across transparent module wrappers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

STAGE2_PARAMETER_NAME_API_VERSION = "longlive_stage2_parameter_names/v1"


def _validated_names(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of parameter names")
    names = tuple(values)
    invalid = [
        name
        for name in names
        if not isinstance(name, str) or not name or name != name.strip()
    ]
    if invalid:
        raise ValueError(f"{label} contains invalid parameter names: {invalid}")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate parameter names")
    return names


def map_parameter_names_to_expected(
    actual_names: Iterable[str],
    expected_names: Iterable[str],
    *,
    label: str,
    require_complete: bool = True,
) -> OrderedDict[str, str]:
    """Map runtime FQNs to one immutable namespace without guessing.

    A runtime name may equal an expected name or add one or more complete
    dotted wrapper segments in front of it.  Zero matches, multiple matches,
    and two runtime names resolving to the same expected name are rejected.
    """

    actual = _validated_names(actual_names, label=f"{label} runtime names")
    expected = _validated_names(expected_names, label=f"{label} expected names")
    if not expected:
        raise ValueError(f"{label} expected parameter names are empty")
    expected_set = set(expected)

    mapped: dict[str, str] = {}
    owners: dict[str, str] = {}
    for actual_name in actual:
        suffixes = [actual_name]
        suffixes.extend(
            actual_name[index + 1 :]
            for index, character in enumerate(actual_name)
            if character == "."
        )
        matches = [suffix for suffix in suffixes if suffix in expected_set]
        if len(matches) != 1:
            raise ValueError(
                f"{label} parameter {actual_name!r} did not map uniquely to the "
                f"expected namespace: matches={matches}"
            )
        expected_name = matches[0]
        if expected_name in owners:
            raise ValueError(
                f"{label} parameter-name collision for {expected_name!r}: "
                f"{owners[expected_name]!r} and {actual_name!r}"
            )
        mapped[actual_name] = expected_name
        owners[expected_name] = actual_name

    if require_complete and set(owners) != expected_set:
        raise ValueError(
            f"{label} parameter-name mapping is incomplete: "
            f"missing={sorted(expected_set - set(owners))}, "
            f"extra={sorted(set(owners) - expected_set)}"
        )
    return OrderedDict((name, mapped[name]) for name in sorted(mapped))


__all__ = [
    "STAGE2_PARAMETER_NAME_API_VERSION",
    "map_parameter_names_to_expected",
]
