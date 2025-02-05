#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os

from rnns_learn_robust_motor_policies.analysis._dependencies import compute_dependencies


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
from copy import deepcopy
from typing import Any

import equinox as eqx
import jax
import jax.random as jr
import jax.tree as jt
import optax 
import plotly

import feedbax
from jax_cookbook import is_type, is_module
import jax_cookbook.tree as jtree  

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.analysis.state_utils import get_pos_endpoints
from rnns_learn_robust_motor_policies.colors import (
    COLORSCALES, 
    get_colors_dicts,
    get_colors_dicts_from_discrete,
)
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.database import add_evaluation, get_db_session
from rnns_learn_robust_motor_policies.hyperparams import TreeNamespace, flatten_hps, load_hps
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training.post_training import TRAINPAIR_SETUP_FUNCS
from rnns_learn_robust_motor_policies.setup_utils import get_base_reaching_task, query_and_load_model
from rnns_learn_robust_motor_policies.hyperparams import namespace_to_dict
from rnns_learn_robust_motor_policies.types import TrainStdDict

from rnns_learn_robust_motor_policies.analysis import ANALYSIS_SETS

#! TODO: `TrainStdDict` should probably not be hardcoded here? Maybe put in the `setup_tasks_and_models` functions
def load_models(db_session, hps: TreeNamespace):
    train_id = int(hps.load.expt_id)
    setup_task_model_pair = TRAINPAIR_SETUP_FUNCS[train_id]
    
    models_base, model_info, replicate_info, n_replicates_included = jtree.unzip(
        TrainStdDict({
            disturbance_std: query_and_load_model(
                db_session,
                setup_task_model_pair,
                params_query=dict(
                    expt_id=train_id,
                    disturbance_type=str(hps.load.disturbance.type), 
                    disturbance_std=disturbance_std,
                    **hps.load.model,
                    **hps.load.train,
                ),
                noise_stds=dict(
                    feedback=hps.model.feedback_noise_std,
                    motor=hps.model.motor_noise_std,
                ),
                surgeries={
                    ('n_steps',): hps.model.n_steps,
                },  
                exclude_underperformers_by=REPLICATE_CRITERION,
            )
            for disturbance_std in hps.load.disturbance.stds
        })
    )
    
    return models_base, model_info, replicate_info, n_replicates_included


def copy_delattr(obj: Any, *attr_names: str):
    """Return a deep copy of an object, with some attributes removed."""
    obj = deepcopy(obj)
    for attr_name in attr_names:
        delattr(obj, attr_name)
    return hps


def use_load_hps_when_none(hps: TreeNamespace) -> TreeNamespace:
    """Replace any unspecified evaluation params with matching loading (training) params"""
    hps_load = hps.load
    hps_other = copy_delattr(hps, 'load')
    hps = hps_other.update_none_leaves(hps_load)
    hps.load = hps_load  
    return hps


def setup_colors(hps):
    colors = {
        k: get_colors_dicts(v, COLORSCALES[k])
        for k, v in dict(
            trials=range(hps.eval_n),
            disturbance_train_stds=hps.load.disturbance.std,
            disturbance_amplitudes=hps.disturbance.amplitude,
        )
    }
    # discrete_colors = {
    #     k: get_colors_dicts_from_discrete(v, COLORSCALES[k])
    #     for k, v in dict(
    #         fb_pert_vars=pert_var_names,
    #     )
    # }
    discrete_colors = {}
    return colors, discrete_colors


def setup_analysis(
    db_session, 
    hps: TreeNamespace,
):
    # If some config values (other than those under the `load` key) are unspecified, replace them with 
    # respective values from the `load` key
    # e.g. if trained on curl fields and hps.disturbance.type is None, use hps.load.disturbance.type
    hps = use_load_hps_when_none(hps)

    # TODO: Load based on module name (e.g. `"part1.plant_perts"`) rather than id (e.g. `"1-1"`)
    # (and then remove the `ANALYSIS_SETS` constructions in the analysis subpackages)
    setup_func, eval_func, analyses = ANALYSIS_SETS[hps.expt_id]

    # Load models
    models_base, model_info, replicate_info, n_replicates_included = load_models(
        db_session,
        hps,
    )
    
    # Later, use this to access the values of hyperparameters, assuming they 
    # are shared between models (e.g. `model_info_0.n_steps`)
    model_info_0 = model_info[hps.load.disturbance.stds[0]]
    
    # If there is no system noise (i.e. the stds are zero), set the number of evaluations per condition to zero.
    # (Is there any other reason than the noise samples, why evaluations might differ?)
    # TODO: Make this optional? 
    #? What is the point of using `jt.leaves` here? 
    any_system_noise = any(jt.leaves((
        hps.model.feedback_noise_std,
        hps.model.motor_noise_std,
    )))
    if not any_system_noise:
        hps.eval_n = 1

    # Get indices for taking important subsets of replicates
    best_replicate, included_replicates = jtree.unzip(TrainStdDict({
        std: (
            replicate_info[std]['best_replicates'][REPLICATE_CRITERION],
            replicate_info[std]['included_replicates'][REPLICATE_CRITERION],
        ) 
        for std in hps.load.disturbance.stds
    }))
    
    # Add evaluation record to the database
    eval_info = add_evaluation(
        db_session,
        expt_id=str(hps.expt_id),
        models=model_info,
        #? Could move the flattening/conversion to `database`?
        #? i.e. everything 
        eval_parameters=namespace_to_dict(flatten_hps(hps)),
        version_info=version_info,
    )
    
    def get_task_variant(models_base, **kwargs):
        task_variant = get_base_reaching_task(
            n_steps=hps.model.n_steps, 
            **kwargs,
        )
    
        tasks, models = setup_func(task_variant, models_base, hps)
        
        return tasks, models
    
    # Outer level is task variants, inner is the structure returned by `setup_func`
    # i.e. "task variants" are a way to evaluate different task setups
    all_tasks, all_models, hps = jtree.unzip({
        k: get_task_variant(models_base, **task_params)
        for k, task_params in namespace_to_dict(hps.task).items()
    })
    
    # And for convenience
    example_task = {
        key: jt.leaves(tasks_variant, is_leaf=is_module)[0]
        for key, tasks_variant in all_tasks.items()
    }
    example_trial_specs = jt.map(lambda task: task.validation_trials, example_task, is_leaf=is_module)
    example_pos_endpoints = jt.map(get_pos_endpoints, example_trial_specs, is_leaf=is_module)
    
    colors, discrete_colors = setup_colors(hps) 
    
    def evaluate_all_states(all_tasks, all_models, hps):
        return {  # Map over task variants
            task_variant_label: jt.map(  # Map over task-modelsubtree pairs generated by `schedule_intervenor` for each base task
                lambda task, models: jt.map(  # Map over base model subtree, for the given base task
                    lambda model: eval_func(model, task, hps, key_eval),
                    models,
                    is_leaf=is_module,
                ),
                all_tasks[task_variant_label],
                all_models[task_variant_label],               
                is_leaf=is_module,
            )
            for task_variant_label in all_tasks
        }

    all_states_bytes = jtree.struct_bytes(eqx.filter_eval_shape(
        evaluate_all_states, all_tasks, all_models, hps
    ))
    print(f"\n{all_states_bytes / 1e9:.2f} GB of memory estimated to store all states.")
    
    all_states = evaluate_all_states(all_tasks, all_models, hps)
    
    #! TODO: This should be mapped over `all_states`, if necessary
    # Each value is a function that is passed `all_states` and returns some data
    # e.g. the result of `"aligned_vars": get_aligned_vars` will be passed to *all* analyses
    # This ensures that dependencies are only calculated once
    dependencies = compute_dependencies(analyses, all_states, hps)
    
    all_results = jt.map(
        lambda analysis: analysis(all_states, hps, **dependencies),
        analyses, 
        is_leaf=is_module,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    args = parser.parse_args()
    
    hps = load_hps(args.config_path)
    
    version_info = log_version_info(
        jax, eqx, optax, plotly, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )

    db_session = get_db_session()
    
    key = jr.PRNGKey(PROJECT_SEED)
    key_init, key_train, key_eval = jr.split(key, 3)
    
    
    