
from collections import namedtuple
from collections.abc import Callable
from typing import Literal, Optional

import equinox as eqx
import jax.numpy as jnp 
import jax.random as jr
import jax.tree as jt

from feedbax import get_ensemble, tree_unzip, make_named_dict_subclass
from feedbax.intervene import (
    CurlField, 
    CurlFieldParams, 
    FixedField, 
    FixedFieldParams,
    schedule_intervenor,
)
from feedbax.task import SimpleReaches, TrialSpecDependency
from feedbax.xabdeef.models import point_mass_nn
from feedbax.xabdeef.losses import simple_reach_loss


INTERVENOR_LABEL = "DisturbanceField"
DISTURBANCE_CLASSES = {
    'curl': CurlField,
    'random': FixedField,
}


TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])
TrainStdDict = make_named_dict_subclass('TrainStdDict')


def get_field_amplitude(intervenor_params):
    if isinstance(intervenor_params, FixedFieldParams):
        return jnp.linalg.norm(intervenor_params.field, axis=-1)
    elif isinstance(intervenor_params, CurlFieldParams):
        return jnp.abs(intervenor_params.amplitude)
    else:
        raise ValueError(f"Unknown intervenor parameters type: {type(intervenor_params)}")


def vector_with_gaussian_length(trial_spec, key):
    key1, key2 = jr.split(key)
    
    angle = jr.uniform(key1, (), minval=-jnp.pi, maxval=jnp.pi)
    length = jr.normal(key2, ())

    return length * jnp.array([jnp.cos(angle), jnp.sin(angle)]) 


disturbance_params = {
    'curl': dict(amplitude=lambda trial_spec, key: jr.normal(key, (1,))),
    'random': dict(field=vector_with_gaussian_length),
}


# Define whether the disturbance is active on each trial
disturbance_active: dict[str, Callable] = {
    "active": lambda p: lambda trial_spec, key: jr.bernoulli(key, p=p),
    "amplitude": lambda p: True,  # All trials perturbed
    "std": lambda p: True,  # All trials perturbed
}


# Define how the network's context input will be determined from the trial specs, to which it is then added
CONTEXT_INPUT_FUNCS = {
    "active": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].active.astype(float),
    "amplitude": lambda trial_specs, key: get_field_amplitude(trial_specs.intervene[INTERVENOR_LABEL]),
    "std": lambda trial_specs, key: trial_specs.intervene[INTERVENOR_LABEL].scale,
}


"""Either scale the field strength by a constant std, or sample the std for each trial"""
SCALE_FUNCS = {
    "active": lambda field_std: field_std,
    "amplitude": lambda field_std: field_std,
    "std": lambda field_std: (
        lambda trial_spec, key: field_std * jnp.abs(jr.normal(key, (1,)))
    ),
}


def setup_task_model_pairs(
    *,
    n_replicates,
    dt,
    mass,
    hidden_size,
    n_steps,
    workspace,
    feedback_delay_steps,
    feedback_noise_std,
    motor_noise_std,
    disturbance_type: Literal['random', 'curl'],
    disturbance_stds,
    p_perturbed,
    key,
):
    """Returns a skeleton PyTree for reloading trained models."""
    task_base = SimpleReaches(
        loss_func=simple_reach_loss(),
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=2,
        eval_n_directions=8,
        eval_reach_length=0.5,    
    )
    
    models_base = get_ensemble(
        point_mass_nn,
        task_base,
        n_extra_inputs=1,  # Contextual input
        n_ensemble=n_replicates,
        dt=dt,
        mass=mass,
        hidden_size=hidden_size, 
        n_steps=n_steps,
        feedback_delay_steps=feedback_delay_steps,
        feedback_noise_std=feedback_noise_std,
        motor_noise_std=motor_noise_std,
        key=key,
    )
    
    def disturbance(field_std, scale_func, active):
        return DISTURBANCE_CLASSES[disturbance_type].with_params(
            scale=scale_func(field_std),
            active=active,
            **disturbance_params[disturbance_type],
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
                    disturbance_std, 
                    SCALE_FUNCS[method_label], 
                    disturbance_active[method_label](p_perturbed),
                ),
                label=INTERVENOR_LABEL,
                default_active=False,
            ))
            for disturbance_std in disturbance_stds
        })
        for method_label, task in tasks.items()
    }
    
    return task_model_pairs



def setup_models(**kwargs):
    task_model_pairs = setup_task_model_pairs(**kwargs)
    _, models = tree_unzip(task_model_pairs)
    return models