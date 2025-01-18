
from collections.abc import Callable, Sequence
from typing import Literal, Optional, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import jax.tree as jt
from jaxtyping import PRNGKeyArray

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
from rnns_learn_robust_motor_policies.types import TaskModelPair, TrainStdDict, TrainingMethodDict


TrainingMethodLabel: TypeAlias = Literal["bcs", "dai", "pai-asf", "pai-n"]

# Separate this def by training method so that we can multiply by `field_std` in the "pai-asf" case,
# without it affecting the context input. That is, in all three cases `field_std` is a factor of 
# the actual field strength, but in `"bcs"` and `"dai"` it is multiplied by the 
# `scale` parameter, which is not seen by the network in those cases; and in `"pai-asf"` it is
# multiplied by the `field` parameter, which is not seen by the network in that case. 
# (See the definition of `SCALE_FUNCS` below.)
disturbance_params = TrainingMethodDict({
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
disturbance_active: dict[str, Callable] = TrainingMethodDict({
    "bcs": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),
    "dai": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
    "pai-asf": lambda p: lambda trial_spec, _, key: jr.bernoulli(key, p=p),  
})


# Define how the network's context input will be determined from the trial specs, to which it is then added
CONTEXT_INPUT_FUNCS = TrainingMethodDict({
    "bcs": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].active.astype(float),
    "dai": lambda trial_specs, key: get_field_amplitude(trial_specs.intervene[INTERVENOR_LABEL]),
    "pai-asf": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].scale,
})


"""Either scale the field strength by a constant std, or sample the std for each trial.

Note that in the `"pai-asf"` case the actual field amplitude is still scaled by `field_std`, 
but this is done in `disturbance_params` so that the magnitude of the context input 
is the same on average between the `"dai"` and `"pai-asf"` methods.
"""
SCALE_FUNCS = TrainingMethodDict({
    "bcs": lambda field_std: field_std,
    "dai": lambda field_std: field_std,
    "pai-asf": lambda field_std: (
        lambda trial_spec, _, key: jr.uniform(key, (), minval=0, maxval=1)
    ),
})


def disturbance(disturbance_type, field_std, p_perturbed, method):
    return DISTURBANCE_CLASSES[disturbance_type].with_params(
        scale=SCALE_FUNCS[method](field_std),
        active=disturbance_active[method](p_perturbed[method]),
        **disturbance_params[method](
            # TODO: Scaleup
            # partial(batch_scale_up, intervention_scaleup_batches[0], n_batches_scaleup)
            field_std
        )[disturbance_type],
    )


def setup_task_model_pair(
    training_method: Optional[TrainingMethodLabel] = None, 
    *,
    n_replicates,
    dt,
    hidden_size,
    n_steps,
    feedback_delay_steps,
    feedback_noise_std,
    motor_noise_std,
    disturbance_type: Literal['constant', 'curl'],
    disturbance_std,
    intervention_scaleup_batches: tuple[int, int],
    p_perturbed,
    key: PRNGKeyArray,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""
    
    # TODO: Implement scale-up 
    n_batches_scaleup = intervention_scaleup_batches[1] - intervention_scaleup_batches[0]
    if n_batches_scaleup > 0:
        def batch_scale_up(batch_start, n_batches, batch_info, x):
            progress = jax.nn.relu(batch_info.current - batch_start) / n_batches
            progress = jnp.minimum(progress, 1.0)
            scale = 0.5 * (1 - jnp.cos(progress * jnp.pi)) 
            return x * scale
    else:
        def batch_scale_up(batch_start, n_batches, batch_info, x):
            return x

    task_base = get_base_task(n_steps=n_steps)
    
    models_base = get_ensemble(
        point_mass_nn,
        task_base,
        n_extra_inputs=1,  # Contextual input
        n=n_replicates,
        dt=dt,
        mass=MASS,
        hidden_size=hidden_size, 
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
        feedback_noise_std=feedback_noise_std,
        motor_noise_std=motor_noise_std,
        key=key,
    )
    
    if training_method is not None:
        task = eqx.tree_at(
            lambda task: task.input_dependencies,
            task_base,
            dict(context=TrialSpecDependency(CONTEXT_INPUT_FUNCS[training_method]))
        )
    else:
        #? In what context do we end up here?
        task = task_base
    
    return TaskModelPair(*schedule_intervenor(
        task, models_base,
        lambda model: model.step.mechanics,
        disturbance(
            disturbance_type,
            disturbance_std, 
            p_perturbed,
            training_method,
        ),
        label=INTERVENOR_LABEL,
        default_active=False,
    ))
