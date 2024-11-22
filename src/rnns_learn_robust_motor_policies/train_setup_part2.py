
from collections.abc import Callable
from typing import Literal, Optional

import equinox as eqx
import jax.numpy as jnp 
import jax.random as jr
import jax.tree as jt

from feedbax import get_ensemble, tree_unzip
from feedbax.intervene import schedule_intervenor
from feedbax.task import SimpleReaches, TrialSpecDependency
from feedbax.xabdeef.models import point_mass_nn
from feedbax.xabdeef.losses import simple_reach_loss

from rnns_learn_robust_motor_policies.misc import vector_with_gaussian_length, get_field_amplitude
from rnns_learn_robust_motor_policies.constants import (
    DISTURBANCE_CLASSES,
    INTERVENOR_LABEL, 
    MASS, 
    WORKSPACE,
)
from rnns_learn_robust_motor_policies.setup_utils import get_base_task
from rnns_learn_robust_motor_policies.types import TaskModelPair, TrainStdDict


# Separate this def by training method so that we can multiply by `field_std` in the "std" case,
# without it affecting the context input. That is, in all three cases `field_std` is a factor of 
# the actual field strength, but in `"active"` and `"amplitude"` it is multiplied by the 
# `scale` parameter, which is not seen by the network in those cases; and in `"std"` it is
# multiplied by the `field` parameter, which is not seen by the network in that case. 
# (See the definition of `SCALE_FUNCS` below.)
disturbance_params = {
    "active": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: jr.normal(key, ()),
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: vector_with_gaussian_length(key),
        ),
    },
    "amplitude": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: jr.normal(key, ()),
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: vector_with_gaussian_length(key),
        ),
    },
    "std": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: field_std * jr.normal(key, ())
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: (
                field_std * vector_with_gaussian_length(key),
            )
        ),
    },
}


# Define whether the disturbance is active on each trial
disturbance_active: dict[str, Callable] = {
    "active": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),
    "amplitude": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
    "std": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
}


# Define how the network's context input will be determined from the trial specs, to which it is then added
CONTEXT_INPUT_FUNCS = {
    # "active": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].active.astype(float),
    # "amplitude": lambda trial_specs, key: get_field_amplitude(trial_specs.intervene[INTERVENOR_LABEL]),
    "std": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].scale,
}


"""Either scale the field strength by a constant std, or sample the std for each trial.

Note that in the `"std"` case the actual field amplitude is still scaled by `field_std`, 
but this is done in `disturbance_params` so that the magnitude of the context input 
is the same on average between the `"amplitude"` and `"std"` methods.
"""
SCALE_FUNCS = {
    "active": lambda field_std: field_std,
    "amplitude": lambda field_std: field_std,
    "std": lambda field_std: (
        lambda trial_spec, _, key: jnp.abs(jr.normal(key, ()))
    ),
}


def disturbance(disturbance_type, field_std, p_perturbed, method):
    return DISTURBANCE_CLASSES[disturbance_type].with_params(
        scale=SCALE_FUNCS[method](field_std),
        active=disturbance_active[method](p_perturbed[method]),
        **disturbance_params[method](field_std)[disturbance_type],
    )


def setup_task_model_pairs(
    *,
    n_replicates,
    dt,
    hidden_size,
    n_steps,
    feedback_delay_steps,
    feedback_noise_std,
    motor_noise_std,
    disturbance_type: Literal['constant', 'curl'],
    disturbance_stds,
    p_perturbed,
    key,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""
    task_base = get_base_task(n_steps)
    
    models_base = get_ensemble(
        point_mass_nn,
        task_base,
        n_extra_inputs=1,  # Contextual input
        n_ensemble=n_replicates,
        dt=dt,
        mass=MASS,
        hidden_size=hidden_size, 
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
        feedback_noise_std=feedback_noise_std,
        motor_noise_std=motor_noise_std,
        key=key,
    )
        
    tasks = {
        method_label: eqx.tree_at(
            lambda task: task.input_dependencies,
            task_base,
            dict(context=TrialSpecDependency(context_input_func))
        )
        for method_label, context_input_func in CONTEXT_INPUT_FUNCS.items()
    }
    
    task_model_pairs = {
        method_label: TrainStdDict({
            disturbance_std: TaskModelPair(*schedule_intervenor(
                task, models_base,
                lambda model: model.step.mechanics,
                disturbance(
                    disturbance_type,
                    disturbance_std, 
                    p_perturbed,
                    method_label,
                ),
                label=INTERVENOR_LABEL,
                default_active=False,
            ))
            for disturbance_std in disturbance_stds
        })
        for method_label, task in tasks.items()
    }
    
    return task_model_pairs
