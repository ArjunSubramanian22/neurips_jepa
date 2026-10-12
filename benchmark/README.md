"""How to audit a checkpoint without copying the metrics.

The numbers are computed by ``src.envs.fork_audit`` and ``src.metrics.cascade``.
``benchmark`` only adapts an environment and a checkpoint onto one JSON file.
Existing trainers and ``scripts/eval_arm_checkpoints.py`` are unchanged.

## Result schema

``schema_version`` is ``counterfactual_benchmark_v1``. Every audit and plan
file has the same header:

- ``command``: ``audit`` or ``plan``
- ``env``, ``model``, ``checkpoint``
- ``split``, ``split_role``, ``is_test_set``
- ``seed``
- ``results``: one entry per task

Each result has ``task``, ``horizon``, ``restoration_exact``, and ``layers``.
The layers, in order, are ``physical_response``, ``observation_visibility``,
``target_response``, ``predictor_response``, ``direction_metrics``,
``ranking_metrics``, ``multi_step_metrics``, and ``planning``. Every layer
has ``status``. ``planning.oracle`` and ``planning.learned`` have their own
status. A tie chooses nothing (``ambiguous_tie``). A missing checkpoint is
``not_measured``, not a zero. Non-finite floats are refused.

``split: test`` is valid only when ``is_test_set`` is true. The arm declares
``train`` and ``val``. The box pusher declares ``live``. Neither has a test
split, so ``--split test`` exits with an error instead of scoring validation.

## Commands

Audit the JEPA arm checkpoint on the validation split:

```bash
python -m benchmark.audit --config configs/benchmark/arm_jepa_audit.yaml
```

The same flags work without a config file:

```bash
python -m benchmark.audit \
    --env mujoco_arm \
    --model jepa \
    --checkpoint outputs/mujoco_arm_runs/jepa/last.pt \
    --split val \
    --task push_object \
    --output experiments/runs/mujoco_arm_v1/benchmark_jepa_push.json
```

Planning uses the same file shape:

```bash
python -m benchmark.plan --config configs/benchmark/arm_jepa_plan.yaml
```

Omit ``--model`` and ``--checkpoint`` to record the simulator oracle only.

## Add a model

Implement ``encode``, ``predict``, ``rollout``, and ``metadata``. ``predict``
and ``rollout`` receive raw environment actions; the adapter normalizes them
with the checkpoint's training split. The default fork target is
``encode`` of the final frame. Override ``represent_fork(action, frames)``
when that target is the wrong object, as it is for the RSSM: the filtered
posterior needs the whole frame sequence, and encoding the last frame alone
restarts the recurrent state.

```python
class MyAdapter:
    def encode(self, obs):
        return latent_vector

    def predict(self, latent, action):
        return next_latent

    def rollout(self, latent, actions):
        return predicted_sequence  # shape (K, D)

    def metadata(self):
        return {"architecture": "my_model"}
```

Register the class name in ``benchmark.adapters.ARCHITECTURES`` and load it
with ``load_adapter``. The evaluator calls ``represent_fork`` when it exists
and otherwise builds the fork from ``encode`` and ``rollout``.

## Add an environment

Wrap a simulator that already has ``get_state`` and ``set_state``:

```python
from benchmark.environments import ForkableEnvironment, register_env, register_task_spec

register_env(
    "my_env",
    lambda task: ForkableEnvironment(MySim(task)),
    ("reach",),
    {"val": ("validation", False)},
)

def spec(task):
    return {
        "actions": {"along": along, "hold": hold, "miss": miss},
        "positive": "along",
        "baseline": "hold",
        "alternate": "miss",
        "horizon": 4,
        "goal_cost": lambda name, future: distance_to_goal(future),
        "learned_reason": "no checkpoint was attached",
    }

register_task_spec("my_env", spec)
```

Declare a test split only if those episodes were never used for training or
model selection. The wrapper's ``clone_state`` and ``restore_state`` call
``get_state`` and ``set_state``.
