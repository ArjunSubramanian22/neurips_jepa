"""Named MLS-MPM environments. ``mpm_legacy`` is the historical dump setting.

Other regimes are calibration candidates or configs written after the
excitation sweep. They do not replace ``MPMConfig()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from scripts.generate_trajectories import MPMConfig


@dataclass(frozen=True)
class EnvironmentSpec:
    """Integrator settings plus how often a command is resampled."""

    environment_id: str
    role: str
    mpm: MPMConfig
    command_hold: int = 3
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.mpm)
        payload.update(
            {
                "environment_id": self.environment_id,
                "role": self.role,
                "command_hold": int(self.command_hold),
                "notes": self.notes,
            }
        )
        return payload


def mpm_legacy() -> EnvironmentSpec:
    """Historical dump: gain 55, radius 0.10, damping 0.08, hold 3, normalized actuator."""
    return EnvironmentSpec(
        environment_id="mpm_legacy",
        role="baseline",
        mpm=MPMConfig(),
        command_hold=3,
        notes=(
            "Exact historical integrator defaults. Command resampling every "
            "3 observation frames matches generate_episode(hold=3)."
        ),
    )


def environment_from_dict(payload: dict[str, Any]) -> EnvironmentSpec:
    """Rebuild a spec from a flat dict. Unknown keys are rejected."""
    mpm_fields = set(MPMConfig.__dataclass_fields__)
    extra = {"environment_id", "role", "command_hold", "notes"}
    unknown = set(payload) - mpm_fields - extra
    if unknown:
        raise ValueError(f"Unknown environment fields: {sorted(unknown)}")
    mpm_kwargs = {key: payload[key] for key in mpm_fields if key in payload}
    return EnvironmentSpec(
        environment_id=str(payload["environment_id"]),
        role=str(payload.get("role", "candidate")),
        mpm=MPMConfig(**mpm_kwargs),
        command_hold=int(payload.get("command_hold", 3)),
        notes=str(payload.get("notes", "")),
    )


def candidate_id(params: dict[str, Any]) -> str:
    """Stable id for one sweep point. Not used for ``mpm_legacy``."""
    return (
        f"cand_d{float(params['damping']):g}"
        f"_g{float(params['action_gain']):g}"
        f"_r{float(params['contact_radius']):g}"
        f"_h{int(params['command_hold'])}"
        f"_w{params['contact_weight_mode']}"
        f"_c{float(params['center_influence']):g}"
        f"_f{float(params['force_influence']):g}"
    )


def calibration_candidates(config: dict[str, Any]) -> list[EnvironmentSpec]:
    """Legacy baseline plus the predeclared grid. Duplicate legacy physics is skipped."""
    legacy = mpm_legacy()
    specs = [legacy]
    seen = {legacy.environment_id}
    grid = config["grid"]
    keys = (
        "damping",
        "action_gain",
        "contact_radius",
        "command_hold",
        "contact_weight_mode",
        "center_influence",
        "force_influence",
    )
    points: list[dict[str, Any]] = []
    axes = [list(grid[key]) for key in keys]
    from itertools import product

    for combo in product(*axes):
        points.append(dict(zip(keys, combo, strict=True)))
    points.extend(dict(item) for item in config.get("extra", []))
    for params in points:
        spec = with_overrides(legacy, candidate_id(params), role="candidate", **params)
        if spec.environment_id in seen:
            continue
        if spec.mpm == legacy.mpm and spec.command_hold == legacy.command_hold:
            continue
        seen.add(spec.environment_id)
        specs.append(spec)
    return specs


def with_overrides(base: EnvironmentSpec, environment_id: str, **overrides: Any) -> EnvironmentSpec:
    """Copy ``base`` and replace MPM or hold fields. ``command_hold`` is not an MPM field."""
    hold = int(overrides.pop("command_hold", base.command_hold))
    role = str(overrides.pop("role", "candidate"))
    notes = str(overrides.pop("notes", ""))
    mpm = replace(base.mpm, **overrides)
    return EnvironmentSpec(
        environment_id=environment_id,
        role=role,
        mpm=mpm,
        command_hold=hold,
        notes=notes,
    )
