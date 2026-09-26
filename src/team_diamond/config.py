"""Typed access to ``configs/*.yaml``, where ``null`` means "not measured yet".

Layer: configuration
See AGENTS.md §15 (Configuration), §16 (Experiment discipline)

The rule this module enforces
-----------------------------
AGENTS.md §15 is explicit that the ``null`` values in the config files are
*placeholders, not defaults*:

    "These ``null`` values are **placeholders, not defaults.** Exact values must
    be determined by experiment."

That is easy to state and easy to forget, because ``yaml.safe_load`` turns
``n_estimators: null`` into a perfectly usable ``None`` that a library will
happily interpret as "use the library default". The result is an experiment
that runs, produces a number, and is attributed to a configuration the team
never actually chose.

So :meth:`Config.require` raises when a value is ``null``, and
:meth:`Config.unmeasured` lists every placeholder in the loaded config. A run
that touches an unmeasured parameter cannot complete by accident.

Layering
--------
``configs/default.yaml`` is loaded first and the specific file is merged over
it, so a file only has to state what it changes. Merging is deep for
dictionaries and replacing for everything else, which means a list in a
specific file replaces the default list rather than appending to it -- appending
would be a silent, order-dependent surprise.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = ["Config", "ConfigError", "UnmeasuredParameter", "config_dir", "load_config"]

T = TypeVar("T")

#: Environment variable pointing at a non-default configs directory.
CONFIG_ROOT_ENV = "TEAM_DIAMOND_CONFIG_ROOT"

#: Default location, relative to the repository root.
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(ValueError):
    """Raised when a required configuration value is missing or ``null``."""


@dataclass(frozen=True, slots=True)
class UnmeasuredParameter:
    """A dotted path whose value is still a placeholder."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path} (still null: {self.reason})"


def config_dir() -> Path:
    """Locate the ``configs/`` directory, verified to exist.

    Returns:
        The directory containing the YAML config files.

    Raises:
        ConfigError: If no configs directory can be found. Failing here beats
            silently falling back to in-code defaults, which is the exact
            failure mode AGENTS.md §15 exists to prevent.
    """
    override = os.environ.get(CONFIG_ROOT_ENV)
    candidates = [Path(override)] if override else []
    candidates.append(_DEFAULT_CONFIG_DIR)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.yaml")):
            return candidate
    raise ConfigError(
        f"no configs directory with YAML files found; tried: "
        f"{[str(c) for c in candidates]}. Set {CONFIG_ROOT_ENV} to point at one."
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` over ``base``, recursing into dictionaries only."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            # Lists replace rather than concatenate. Concatenating would make
            # the result depend on file load order, which is not a property
            # anyone wants in a reproducibility-critical artefact.
            merged[key] = value
    return merged


def _null_paths(value: Any, prefix: str = "") -> Iterator[UnmeasuredParameter]:
    """Yield a placeholder record for every ``null`` leaf in a config tree."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _null_paths(child, f"{prefix}.{key}" if prefix else str(key))
    elif value is None:
        reason = "a value has not been measured yet"
        yield UnmeasuredParameter(path=prefix, reason=reason)


@dataclass(frozen=True, slots=True)
class Config:
    """A merged, read-only configuration tree.

    Use :func:`load_config` rather than constructing this directly.
    """

    data: Mapping[str, Any]
    sources: tuple[Path, ...]

    # -- access ---------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, returning ``default`` if absent.

        Args:
            dotted: Path such as ``"model.params.random_state"``.
            default: Returned when the path is missing.

        Returns:
            The value at that path, or ``default``.

        Raises:
            ConfigError: If an intermediate component exists but is not a
                mapping, which means the path is wrong rather than absent.
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping):
                raise ConfigError(
                    f"config path {dotted!r} is invalid: {part!r} was reached "
                    f"inside a {type(node).__name__}, not a section"
                )
            if part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Fetch a value that must have been measured.

        Args:
            dotted: Path such as ``"model.params.n_estimators"``.

        Returns:
            The value, guaranteed non-``None``.

        Raises:
            ConfigError: If the path is absent, ``None``, or empty. The message
                names the path and says it is a placeholder, so the fix is
                obvious: measure it, or choose a value and write it down.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(
                f"config key {dotted!r} is absent from "
                f"{[p.name for p in self.sources]}. Add it explicitly; an "
                f"absent key is not a default."
            )
        if value is None:
            raise ConfigError(
                f"config key {dotted!r} is null. In this project a null means "
                f"'PROPOSAL - not measured yet' (AGENTS.md §15), not 'use the "
                f"library default'. Measure it, or record a chosen value in "
                f"the config file."
            )
        if isinstance(value, (str, list, dict, tuple)) and len(value) == 0:
            raise ConfigError(
                f"config key {dotted!r} is empty. Treat that as unmeasured "
                f"rather than as a deliberate zero."
            )
        return value

    def section(self, dotted: str) -> Mapping[str, Any]:
        """Fetch a whole section as a mapping.

        Args:
            dotted: Path to the section.

        Returns:
            The mapping. Empty when the path is absent.

        Raises:
            ConfigError: If the value at that path is not a mapping.
        """
        value = self.get(dotted, {})
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"config path {dotted!r} is a {type(value).__name__}, "
                f"expected a section"
            )
        return value

    # -- reporting ------------------------------------------------------------
    def unmeasured(self) -> tuple[UnmeasuredParameter, ...]:
        """Every ``null`` placeholder in the merged config.

        Returns:
            One record per unmeasured parameter, sorted by path.

        Notes:
            This is the list to paste into a pull request or an experiment
            record. It is the honest answer to "what in here is still a guess".
        """
        return tuple(sorted(_null_paths(self.data), key=lambda p: p.path))

    def require_all_measured(self) -> None:
        """Raise if *any* parameter in the config is still a placeholder.

        Call this at the top of a production or leaderboard run, where every
        number should be a decision the team made rather than a library
        default. An experiment sweep will legitimately have unmeasured entries,
        so this is opt-in rather than automatic.
        """
        pending = self.unmeasured()
        if pending:
            listing = "\n".join(f"  - {p}" for p in pending)
            raise ConfigError(
                f"{len(pending)} configuration parameter(s) are still "
                f"unmeasured placeholders:\n{listing}\n\n"
                f"Measure them, or record an explicit choice, before treating "
                f"this run as a result."
            )

    def describe(self) -> str:
        """Render the config as YAML, with a header noting the source files."""
        import yaml

        header = "# merged from: " + ", ".join(p.name for p in self.sources)
        return header + "\n" + yaml.safe_dump(dict(self.data), sort_keys=False)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """Return a new config with ``overrides`` merged over the top.

        Used for command-line overrides that must be *recorded*, not silently
        applied, so an experiment is reproducible from its log.

        Args:
            overrides: Nested mapping, same shape as the config tree.

        Returns:
            A new :class:`Config`.
        """
        return Config(
            data=_deep_merge(self.data, overrides), sources=self.sources
        )


def load_config(*names: str) -> Config:
    """Load ``default.yaml`` and merge the named files over it.

    Args:
        *names: File names without extension, e.g. ``"model"``, ``"features"``.
            ``"default"`` is always loaded first; naming it again is harmless.

    Returns:
        The merged configuration.

    Raises:
        ConfigError: If a named file does not exist, or ``default.yaml`` is
            missing. A silently-ignored config file would make an experiment
            irreproducible in the worst possible way: it would look
            reproducible.
    """
    directory = config_dir()
    base = directory / "default.yaml"
    if not base.is_file():
        raise ConfigError(f"missing required config file: {base}")

    sources: list[Path] = [base]
    merged: dict[str, Any] = _read_yaml(base)

    for name in names:
        path = directory / f"{name}.yaml"
        if not path.is_file():
            raise ConfigError(
                f"config file not found: {path}\nAvailable: "
                f"{sorted(p.stem for p in directory.glob('*.yaml'))}"
            )
        if path == base:
            continue
        merged = _deep_merge(merged, _read_yaml(path))
        sources.append(path)

    return Config(data=merged, sources=tuple(sources))


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a mapping."""
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, "
            f"found {type(loaded).__name__}"
        )
    return dict(loaded)
