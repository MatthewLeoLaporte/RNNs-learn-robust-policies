from functools import partial
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from feedbax import get_ensemble, is_module
from feedbax.intervene import schedule_intervenor
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.train import filter_spec_leaves
from feedbax.xabdeef.models import point_mass_nn
from feedbax.xabdeef.losses import simple_reach_loss

from rnns_learn_robust_motor_policies.constants import (
    DISTURBANCE_CLASSES, 
    INTERVENOR_LABEL, 
    MASS,
)
from rnns_learn_robust_motor_policies.misc import vector_with_gaussian_length
from rnns_learn_robust_motor_policies.setup_utils import get_base_task, get_train_pairs_by_disturbance_std
from rnns_learn_robust_motor_policies.types import TaskModelPair, TrainStdDict


disturbance_params = lambda scale_func: {
    'curl': dict(
        amplitude=lambda trial_spec, batch_info, key: scale_func(
            batch_info,
            jr.normal(key, ()), 
        )
    ),
    'constant': dict(
        field=lambda trial_spec, batch_info, key: scale_func(
            batch_info,
            vector_with_gaussian_length(key), 
        )
    ),
}


def setup_task_model_pair(
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
    control_loss_scale: float,
    key,
    **kwargs,
):
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
    
    loss_func = simple_reach_loss()

    loss_func = eqx.tree_at(
        lambda loss_func: loss_func.weights['nn_output'],
        loss_func,
        control_loss_scale * loss_func.weights['nn_output'],
    )
    
    task_base = get_base_task(
        n_steps=n_steps,
        loss_func=loss_func,
    )
    
    models = get_ensemble(
        point_mass_nn,
        task_base,
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
    
    def disturbance(field_std, active=True):
        return DISTURBANCE_CLASSES[disturbance_type].with_params(
            scale=field_std,
            active=active,
            **disturbance_params(
                partial(batch_scale_up, intervention_scaleup_batches[0], n_batches_scaleup)
            )[disturbance_type],
        )
        
    return TaskModelPair(*schedule_intervenor(
        task_base, models,
        lambda model: model.step.mechanics,
        disturbance(disturbance_std),
        label=INTERVENOR_LABEL,
        default_active=False,
    ))


def setup_model_parameter_histories(
    models_tree,
    *,
    where_train_strs,
    save_model_parameters,
    key,
):
    n_save_steps = len(save_model_parameters)
    where_train = attr_str_tree_to_where_func(where_train_strs)
    
    models_parameters = jt.map(
        lambda models: eqx.filter(eqx.filter(
            models, 
            filter_spec_leaves(models, where_train),
        ), eqx.is_array),
        models_tree,
        is_leaf=is_module,
    )
    
    model_parameter_histories = jt.map(
        lambda x: (
            jnp.empty((n_save_steps,) + x.shape)
            if eqx.is_array(x) else x
        ),
        models_parameters,
    )
    
    return model_parameter_histories



def get_train_pairs(hps, key):
    """Given hyperparams and a particular task-model pair setup function, return the PyTree of task-model pairs.
    
    Here in Part 1 this is trivial since we're only training a single `TrainStdDict` of models.
    """
    return get_train_pairs_by_disturbance_std(
        setup_task_model_pair, hps['model'], hps['disturbance'], key
    )