#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os

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
from jax_cookbook import is_type
import jax_cookbook.tree as jtree  

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.database import add_evaluation, get_db_session
from rnns_learn_robust_motor_policies.hyperparams import TreeNamespace, flatten_hps, load_hps
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training.post_training import TRAINPAIR_SETUP_FUNCS
from rnns_learn_robust_motor_policies.setup_utils import query_and_load_model
from rnns_learn_robust_motor_policies.hyperparams import namespace_to_dict
from rnns_learn_robust_motor_policies.types import TrainStdDict


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
    
    # Load models
    models_base, model_info, replicate_info, n_replicates_included = load_models(
        db_session,
        hps,
    )
    
    # If some config values (other than those under the `load` key) are unspecified, replace them with 
    # respective values from the `load` key
    # e.g. if trained on curl fields and hps.disturbance.type is None, use hps.load.disturbance.type
    hps = use_load_hps_when_none(hps)
    
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
        hps.eval.n = 1
        hps.eval.n_small = 1

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
    # TODO: Setup the evaluation tasks
    #! Varies with analysis
    
    # TODO: Set up plots, colors
    #! Varies with spreads (train std, eval amp, etc.)
    
    # TODO: Evaluate all states
    
    # TODO: Align variables 
    # optionally? or we could indicate it somehow as a dependency, e.g. 3 analyses depend on the aligned variables,
    # but we should only compute them once
    
    # TODO: Individual analyses
    #! Varies
    