#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os
from types import SimpleNamespace

from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.setup_utils import query_and_load_model
from rnns_learn_robust_motor_policies.training.train import load_hps
from rnns_learn_robust_motor_policies.tree_utils import dict_to_namespace
from rnns_learn_robust_motor_policies.types import TrainStdDict

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse


import equinox as eqx
import jax
import jax.random as jr
import jax.tree as jt
import optax 
import plotly

import feedbax
from feedbax import tree_unzip, is_type

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED

from rnns_learn_robust_motor_policies.database import ModelRecord, add_evaluation, get_db_session, use_record_params_where_none
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training import TRAINPAIR_SETUP_FUNCS


def load_models(db_session, hps: SimpleNamespace):
    train_id = int(hps.load.id)
    setup_task_model_pair = TRAINPAIR_SETUP_FUNCS[train_id]
    
    models_base, model_info, replicate_info, n_replicates_included = tree_unzip(
        TrainStdDict({
            disturbance_std: query_and_load_model(
                db_session,
                setup_task_model_pair,
                params_query=dict(
                    origin=train_id,
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
    
    best_replicate, included_replicates = tree_unzip(TrainStdDict({
        std: (
            replicate_info[std]['best_replicates'][REPLICATE_CRITERION],
            replicate_info[std]['included_replicates'][REPLICATE_CRITERION],
        ) 
        for std in hps['load']['disturbance']['stds']
    }))
    
    # !!!!!!!!!
    # TODO: This depends on the type of analysis; e.g. 1-1 needs `disturbance_type` but 1-2 doesn't
    # TODO: Also, this only contains the parameters listed here?
    # We should probably just do some automatic updating of `hps - hps['load']` with `hps['load']`
    all_eval_parameters = jt.map(
        lambda record: use_record_params_where_none(
            dict(
                # disturbance_type=hps.disturbance.type,
                feedback_noise_std=hps.model.feedback_noise_std,
                motor_noise_std=hps.model.motor_noise_std,
            ), 
            record,
        ),
        model_info,
        is_leaf=is_type(ModelRecord),
    )

    # Check that all the eval parameters are the same, for the different models.
    #! I don't remember the point of this. Shouldn't (for example) `feedback_noise_std` be the same for
    #! 1) all the models we've loaded, because we queried a single value of it, i.e. `hps.load.model.feedback_noise_std`, and
    #! 2) the current notebook (i.e. `hps_ns.model.feedback_noise_std`)?
    #! Thus we shouldn't need the next two lines. Or do we?
    all_eval_params_flat =[tuple(d.items()) for d in all_eval_parameters.values()]
    assert len(set(all_eval_params_flat)) == 1

    eval_parameters = all_eval_parameters[hps.load.disturbance.stds[0]]
    
    # Later, use this to access the values of hyperparameters, assuming they 
    # are shared between models (e.g. `model_info_0.n_steps`)
    model_info_0 = model_info[hps.load.disturbance.stds[0]]
    
    # TODO: Notify the user if the training run has been run before, and is already in the database

    any_system_noise = any(jt.leaves((
        eval_parameters['feedback_noise_std'],
        eval_parameters['motor_noise_std'],
    )))
    
    # TODO: Make this optional
    if not any_system_noise:
        hps.eval.n = 1
        hps.eval.n_small = 1
        
    eval_parameters |= dict(
        disturbance_amplitudes=hps.disturbance.amplitudes,
        n_evals=hps.eval.n,
        n_evals_small=hps.eval.n_small,
    )
    
    # Initialize a record in the evaluations database
    eval_info = add_evaluation(
        db_session,
        origin=str(hps.id),
        models=model_info,
        eval_parameters=eval_parameters,
        version_info=version_info,
    )
    
    # TODO: Setup the evaluation tasks
    
    # TODO: Set up plots, colors
    
    # TODO: Evaluate all states
    
    # TODO: Align variables 
    # optionally? or we could indicate it somehow as a dependency, e.g. 3 analyses depend on the aligned variables,
    # but we should only compute them once
    
    # TODO: 
    