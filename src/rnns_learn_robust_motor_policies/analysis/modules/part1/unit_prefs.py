from functools import partial

import jax.numpy as jnp
import jax.tree as jt

from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_PERT_FUNCS
from rnns_learn_robust_motor_policies.analysis.network import UnitPreferences
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.misc import map_fn_over_tree, vectors_to_2d_angles
from rnns_learn_robust_motor_policies.types import LDict


ID = "1-4"


COLOR_FUNCS = {}


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown perturbation type: {hps.pert.type}")

    # Insert the disturbance field component into each model
    models = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {PLANT_INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )

    # Assume a sequence of amplitudes is provided, as in the default config
    pert_amps = hps.pert.amp
    # Construct tasks with different amplitudes of disturbance field
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda pert_amp: schedule_intervenor(
            task_base, models,
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps))
        ),
    ))

    all_hps = jt.map(lambda _: hps, all_tasks, is_leaf=is_module)

    return all_tasks, all_models, all_hps, None


eval_func = vmap_eval_ensemble


def get_goal_positions(task, states):
    targets = task.validation_trials.targets["mechanics.effector.pos"].value
    return targets[..., -1:, :]


ts = jnp.arange(0, 20)

ALL_ANALYSES = [
    (
        UnitPreferences(feature_fn=get_goal_positions)
        .after_indexing(-2, ts, axis_label="timestep")
        .and_transform_results(map_fn_over_tree(vectors_to_2d_angles))
    ),
]