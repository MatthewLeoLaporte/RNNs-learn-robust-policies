from collections.abc import Callable
from functools import partial
from typing import Literal as L, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.intervene import schedule_intervenor
from feedbax.task import TrialSpecDependency
from feedbax.xabdeef.models import point_mass_nn
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.misc import vector_with_gaussian_length, get_field_amplitude
from rnns_learn_robust_motor_policies.constants import (
    DISTURBANCE_CLASSES,
    INTERVENOR_LABEL, 
    MASS,
)
from rnns_learn_robust_motor_policies.setup_utils import get_base_reaching_task, get_train_pairs_by_pert_std
from rnns_learn_robust_motor_policies.types import TaskModelPair, LDict


TrainingMethodLabel: TypeAlias = L["bcs", "dai", "pai-asf", "pai-n"]


# Separate this def by training method so that we can multiply by `field_std` in the "pai-asf" case,
# without it affecting the context input. That is, in all three cases `field_std` is a factor of 
# the actual field strength, but in `"bcs"` and `"dai"` it is multiplied by the 
# `scale` parameter, which is not seen by the network in those cases; and in `"pai-asf"` it is
# multiplied by the `field` parameter, which is not seen by the network in that case. 
# (See the definition of `SCALE_FUNCS` below.)
disturbance_params = LDict.of("train__method")({
    "bcs": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: jr.normal(key, ()),
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: vector_with_gaussian_length(key),
        ),
    },
    "dai": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: jr.normal(key, ()),
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: vector_with_gaussian_length(key),
        ),
    },
    "pai-asf": lambda field_std: {
        'curl': dict(
            amplitude=lambda trial_spec, batch_info, key: field_std * jr.normal(key, ())
        ),
        'constant': dict(
            field=lambda trial_spec, batch_info, key: (
                field_std * vector_with_gaussian_length(key)
            )
        ),
    },
})


# Define whether the disturbance is active on each trial
disturbance_active: LDict[str, Callable] = LDict.of("train__method")({
    "bcs": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),
    "dai": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
    "pai-asf": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
})


# Define how the network's context input will be determined from the trial specs, to which it is then added
CONTEXT_INPUT_FUNCS = LDict.of("train__method")({
    "bcs": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].active.astype(float),
    "dai": lambda trial_specs, key: get_field_amplitude(trial_specs.intervene[INTERVENOR_LABEL]),
    "pai-asf": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].scale,
})


# TODO: Move to config yaml
P_PERTURBED = LDict.of("train__method")({
    "bcs": 0.5,
    "dai": 1.0,
    "pai-asf": 1.0,
})


"""Either scale the field strength by a constant std, or sample the std for each trial.

Note that in the `"pai-asf"` case the actual field amplitude is still scaled by `field_std`, 
but this is done in `disturbance_params` so that the magnitude of the context input 
is the same on average between the `"dai"` and `"pai-asf"` methods.
"""
SCALE_FUNCS = LDict.of("train__method")({
    "bcs": lambda field_std: field_std,
    "dai": lambda field_std: field_std,
    "pai-asf": lambda field_std: (
        lambda trial_spec, _, key: jr.uniform(key, (), minval=0, maxval=1)
    ),
})


def disturbance(pert_type, field_std, method):
    return DISTURBANCE_CLASSES[pert_type].with_params(
        scale=SCALE_FUNCS[method](field_std),
        active=disturbance_active[method](P_PERTURBED[method]),
        **disturbance_params[method](
            # TODO: Scaleup
            # partial(batch_scale_up, intervention_scaleup_batches[0], n_batches_scaleup)
            field_std
        )[pert_type],
    )


def setup_task_model_pair(
    hps: TreeNamespace = TreeNamespace(),
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""   
    hps = hps | kwargs
    
    # TODO: Implement scale-up for this experiment
    scaleup_batches = hps.train.intervention_scaleup_batches
    n_batches_scaleup = scaleup_batches[1] - scaleup_batches[0]
    if n_batches_scaleup > 0:
        def batch_scale_up(batch_start, n_batches, batch_info, x):
            progress = jax.nn.relu(batch_info.current - batch_start) / n_batches
            progress = jnp.minimum(progress, 1.0)
            scale = 0.5 * (1 - jnp.cos(progress * jnp.pi)) 
            return x * scale
    else:
        def batch_scale_up(batch_start, n_batches, batch_info, x):
            return x

    task_base = get_base_reaching_task(n_steps=hps.model.n_steps)
    
    models_base = jtree.get_ensemble(
        point_mass_nn,
        task_base,
        n_extra_inputs=1,  # Contextual input
        n=hps.model.n_replicates,
        dt=hps.model.dt,
        mass=MASS,
        damping=hps.model.damping,
        hidden_size=hps.model.hidden_size, 
        n_steps=hps.model.n_steps,
        feedback_delay_steps=hps.model.feedback_delay_steps,
        feedback_noise_std=hps.model.feedback_noise_std,
        motor_noise_std=hps.model.motor_noise_std,
        key=key,
    )
    
    try:
        task = eqx.tree_at(
            lambda task: task.input_dependencies,
            task_base,
            dict(context=TrialSpecDependency(CONTEXT_INPUT_FUNCS[hps.train.method]))
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps.train.method")
    
    return TaskModelPair(*schedule_intervenor(
        task, models_base,
        lambda model: model.step.mechanics,
        disturbance(
            hps.train.pert.type,
            hps.train.pert.std, 
            # p_perturbed,
            hps.train.method,
        ),
        label=INTERVENOR_LABEL,
        default_active=False,
    ))


def get_train_pairs(hps: TreeNamespace, key: PRNGKeyArray):
    """Given hyperparams and a particular task-model pair setup function, return the PyTree of task-model pairs."""
    
    get_train_pairs_partial = partial(
        get_train_pairs_by_pert_std, 
        setup_task_model_pair, 
        key=key,  # Use the same PRNG key for all training methods
    )
    
    task_model_pairs, all_hps = jtree.unzip(LDict.of("train__method")({
        method_label: get_train_pairs_partial(hps | dict(train=dict(method=method_label)))
        #! Assume `hps.train.method` is a list of training method labels
        for method_label in hps.train.method
    }))
    
    return task_model_pairs, all_hps