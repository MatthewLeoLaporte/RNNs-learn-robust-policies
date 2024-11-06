from operator import attrgetter
from typing import Literal
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt

from feedbax import get_ensemble, is_module, tree_unzip
from feedbax.intervene import CurlField, FixedField, schedule_intervenor
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.task import SimpleReaches
from feedbax.train import filter_spec_leaves
from feedbax.xabdeef.models import point_mass_nn
from feedbax.xabdeef.losses import simple_reach_loss


def setup_models(
    *,
    n_replicates,
    dt,
    mass,
    hidden_size,
    n_steps,
    feedback_delay_steps,
    feedback_noise_std,
    motor_noise_std,
    disturbance_type: Literal['random', 'curl'],
    disturbance_stds,
    key,
):
    """Returns a skeleton PyTree for reloading trained models."""
    task_train_dummy = SimpleReaches(
        loss_func=simple_reach_loss(), 
        n_steps=n_steps,
        workspace=((0, 0), (0, 0)),
    )
    
    models = get_ensemble(
        point_mass_nn,
        task_train_dummy,
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
    
    if disturbance_type == 'curl':        
        disturbance = CurlField.with_params(amplitude=jnp.array(1).item())    
            
    elif disturbance_type == 'random':
        disturbance = FixedField.with_params(
            scale=1,
            field=jnp.array([1.0, 0.0]),  # to the right
        ) 
    
    _, models = tree_unzip(jt.map(
        lambda field_std: schedule_intervenor(
            task_train_dummy, models,
            lambda model: model.step.mechanics,
            disturbance,
            default_active=False,
        ),
        disturbance_stds,    
    ))
    
    return dict(zip(disturbance_stds, models))


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



