from operator import attrgetter
from typing import Literal
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt

from feedbax import get_ensemble, is_module, tree_unzip
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.intervene import CurlField, FixedField, schedule_intervenor
from feedbax.train import filter_spec_leaves, init_task_trainer_history
from feedbax.xabdeef.models import point_mass_nn
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.task import SimpleReaches


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
        lambda curl_std: schedule_intervenor(
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


def setup_train_histories(
    task,  
    models_tree,
    disturbance_stds,
    n_batches,
    batch_size,
    n_replicates,
    *,
    where_train_strs,
    save_model_parameters,
    key,
):
    """Returns a skeleton PyTree for the training histories (losses, parameter history, etc.)
    
    Note that `init_task_trainer_history` depends on `task` to infer 
    
    1) The number and name of loss function terms;
    2) The structure of trial specs, in case `save_trial_specs is not None`.
    
    Here, neither of these are much of a concern since 1) we are always using the same 
    loss function for each set of saved/loaded models in this project, 2) `save_trial_specs is None`.
    """   
    where_train = attr_str_tree_to_where_func(where_train_strs)
    
    assert list(models_tree.keys()) == list(disturbance_stds)
    
    aaa= {
        train_std: init_task_trainer_history(
            task,
            n_batches,
            n_replicates,
            ensembled=True,
            ensemble_random_trials=False,
            save_model_parameters=jnp.array(save_model_parameters),
            save_trial_specs=None,
            batch_size=batch_size,
            model=model,
            where_train=where_train,  
        )
        for train_std, model in models_tree.items()
    }

    return aaa
