"""Fork a reduced MLS-MPM state and score each causal layer separately.

The learned encoder, learned predictor, and planner are recorded as
``not_measured``. Privileged particle coordinates and two stand-in predictors
are positive controls. They are not AC-JEPA results, and the reduced grid is
not the historical dump.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.generate_trajectories import (
    MLSMPM2D,
    MPMConfig,
    apply_action_frames,
    render_rgb,
    sample_action,
    sample_geometry,
)
from src.metrics.cascade import (
    action_delta,
    assemble_cascade,
    cosine_alignment,
    latent_contrast,
    not_measured,
    observation_visibility,
    physical_response,
)
from src.metrics.ranking import chance_top1, random_ranking_top1

LEARNED_REPRESENTATION = not_measured(
    "no encoder checkpoint is evaluated in cascade_protocol_smoke_v1"
)
LEARNED_PREDICTOR = not_measured(
    "no predictor checkpoint is evaluated in cascade_protocol_smoke_v1"
)
DECISION = not_measured(
    "decision utility is measured in planning_oracle_sanity_v1, not on these forks"
)


def synthetic_controls(seed: int = 0) -> dict[str, Any]:
    """Call the metric functions on inputs whose values are known."""
    action = np.array([0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    latent = np.array([0.25, -0.5, 0.75, 1.0], dtype=np.float64)
    ranking = random_ranking_top1(4000, 4, np.random.default_rng(seed))
    return {
        "identical_action_l2": action_delta(action, action)["l2"],
        "identical_latent_mse": latent_contrast(latent, latent)["mse"],
        "aligned_cosine": cosine_alignment(latent, 3.0 * latent)["cosine"],
        "opposite_cosine": cosine_alignment(latent, -2.0 * latent)["cosine"],
        "random_ranking_top1": ranking["top1"],
        "random_ranking_chance": ranking["chance"],
        "random_ranking_n_trials": ranking["n_trials"],
        "random_ranking_n_hits": ranking["n_hits"],
        "chance_top1_four": chance_top1(4),
    }


def _intervention_action(name: str, factual: np.ndarray) -> np.ndarray:
    action = np.asarray(factual, dtype=np.float32).copy()
    if name == "identical":
        return action
    if name == "force_reversed":
        action = action.copy()
        action[2:4] *= np.float32(-1.0)
        return action
    if name == "zero_force":
        action = action.copy()
        action[2:4] = np.float32(0.0)
        return action
    raise ValueError(f"unknown intervention {name!r}.")


def measure_forks(
    *,
    seed: int,
    n_frames: int,
    youngs: float,
    geometry: str,
    interventions: list[str],
    mpm: MPMConfig,
) -> dict[str, Any]:
    """Roll one factual action and each intervention from the same snapshot."""
    if int(n_frames) < 1:
        raise ValueError("n_frames must be >= 1.")
    rng = np.random.default_rng(int(seed))
    sim = MLSMPM2D(mpm)
    points = sample_geometry(geometry, mpm.particle_spacing, rng)
    sim.reset(points, float(youngs))
    warmup = sample_action(rng, sim.x, mpm.action_dim)
    apply_action_frames(sim, warmup, 1)
    factual = sample_action(rng, sim.x, mpm.action_dim)
    factual_sim = sim.clone()
    apply_action_frames(factual_sim, factual, int(n_frames))
    factual_frame = render_rgb(factual_sim.x, factual_sim.v, mpm.img_size, mpm.render_sigma)
    factual_embedding = factual_sim.x.reshape(-1).astype(np.float64)
    pairs: list[dict[str, Any]] = []
    for name in interventions:
        action = _intervention_action(name, factual)
        branch = sim.clone()
        apply_action_frames(branch, action, int(n_frames))
        frame = render_rgb(branch.x, branch.v, mpm.img_size, mpm.render_sigma)
        embedding = branch.x.reshape(-1).astype(np.float64)
        privileged = latent_contrast(factual_embedding, embedding)
        constant = latent_contrast(factual_embedding, factual_embedding)
        oracle = latent_contrast(factual_embedding, embedding)
        alignment = cosine_alignment(embedding - factual_embedding, embedding - factual_embedding)
        cascade = assemble_cascade(
            action=action_delta(factual, action),
            physical=physical_response(factual_sim.x, branch.x, factual_sim.v, branch.v),
            observation=observation_visibility(factual_frame, frame),
            representation=LEARNED_REPRESENTATION,
            predictor=LEARNED_PREDICTOR,
            decision=DECISION,
        )
        pairs.append(
            {
                "intervention": name,
                "cascade": cascade,
                "controls": {
                    "privileged_particle_coordinates": {
                        "family": "privileged_particle_coordinates",
                        "contrast": privileged,
                        "note": "Particle coordinates are simulator state, not a learned encoder.",
                    },
                    "constant_predictor_standin": {
                        "family": "constant_predictor_standin",
                        "contrast": constant,
                        "alignment": cosine_alignment(
                            np.zeros_like(factual_embedding),
                            embedding - factual_embedding,
                        ),
                        "note": "Emits the factual embedding for every action.",
                    },
                    "oracle_privileged_standin": {
                        "family": "oracle_privileged_standin",
                        "contrast": oracle,
                        "alignment": alignment,
                        "note": "Emits the privileged embedding of the rolled future.",
                    },
                },
            }
        )
    return {
        "seed": int(seed),
        "n_frames": int(n_frames),
        "youngs": float(youngs),
        "geometry": geometry,
        "n_particles": int(factual_sim.x.shape[0]),
        "simulator": {
            "n_grid": int(mpm.n_grid),
            "img_size": int(mpm.img_size),
            "particle_spacing": float(mpm.particle_spacing),
            "substeps": int(mpm.substeps),
            "dt": float(mpm.dt),
            "damping": float(mpm.damping),
            "comparable_to_paper_dump": False,
        },
        "pairs": pairs,
        "synthetic_controls": synthetic_controls(seed),
    }
