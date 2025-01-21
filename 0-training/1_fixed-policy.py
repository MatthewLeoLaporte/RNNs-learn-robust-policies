#!/usr/bin/env python
# coding: utf-8

NB_ID = "1"

## Environment setup

import os
from pathlib import Path

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
from functools import partial

import equinox as eqx
import jax
import jax.random as jr
import jax.tree as jt
from jaxtyping import PRNGKeyArray
import optax 

import feedbax
from feedbax import (
    is_type,
    tree_map_tqdm,
    tree_unzip,
)
from feedbax.misc import attr_str_tree_to_where_func

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.config import load_config, load_default_config
from rnns_learn_robust_motor_policies.database import (
    get_db_session,
)
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.setup_utils import (
    process_hps,
    save_all_models,
)
from rnns_learn_robust_motor_policies.train_setup_part1 import (
    setup_task_model_pair,
)
from rnns_learn_robust_motor_policies.tree_utils import (
    deep_update, 
    map_kwargs_to_dict,
)
from rnns_learn_robust_motor_policies.types import TaskModelPair
from rnns_learn_robust_motor_policies.train_setup import (
    train_pair,
    train_setup,
)
from rnns_learn_robust_motor_policies.types import TrainStdDict


CONFIG_DIR = Path('../config')


def hps_given_path(path, model_hps, train_hps):
    # This is specific to notebook 1.
    return (
        model_hps | dict(disturbance_std=path[0]),
        train_hps,
    )


def construct_model_pytree(model_hps, disturbance, key):
    task_model_pairs = TrainStdDict(map_kwargs_to_dict(
        partial(
            setup_task_model_pair, 
            **model_hps, 
            key=key,
        ),
        'disturbance_std',
        disturbance['stds'][disturbance['type']],  # The sequence of stds for the given disturbance type
    ))
    return task_model_pairs


def main(config: dict, key: PRNGKeyArray):
    key_init, key_train, key_eval = jr.split(key, 3)
    
    model_hps, train_hps, disturbance = process_hps(config)

    # TODO: Refactor this into a function, and then I think the only difference between part 1 and part 2
    # is 1) that function, and 2) `hps_given_path`, if we are using it. All the other differences are taken 
    # care of by `setup_task_model_pair`
    ## Construct a model (ensemble) for each value of the disturbance std
    task_model_pairs = TrainStdDict(map_kwargs_to_dict(
        partial(
            setup_task_model_pair, 
            **model_hps, 
            key=key_init,
        ),
        'disturbance_std',
        disturbance['stds'][disturbance['type']],  # The sequence of stds for the given disturbance type
    ))
    
    trainer, loss_func = train_setup(train_hps)
    
    # Convert string representations of where-functions to actual functions.
    # 
    #   - Strings are easy to serialize, or to specify in config files; functions are not.
    #   - These where-functions are for selecting the trainable nodes in the pytree of model 
    #     parameters.
    #
    where_train = {
        i: attr_str_tree_to_where_func(strs) 
        for i, strs in train_hps['where_train_strs'].items()
    }
    
    ## Train all the models.
    # Organize the constant arguments for the calls to `train_pair`
    train_args = dict(
        ensembled=True,
        loss_func=loss_func,
        # TODO: Is this correct? Or should we pass the task for the respective training method?
        task_baseline=jt.leaves(task_model_pairs, is_leaf=is_type(TrainStdDict))[0][0].task, 
        where_train=where_train,
        batch_size=train_hps['batch_size'], 
        log_step=500,
        save_model_parameters=train_hps['save_model_parameters'],
        state_reset_iterations=train_hps['state_reset_iterations'],
        # disable_tqdm=True,
    )

    # The imported `train_pair` function actually runs the trainer
    trained_models, train_histories = tree_unzip(tree_map_tqdm(
        partial(train_pair, trainer, train_hps['n_batches'], key=key_train, **train_args),
        task_model_pairs,
        label="Training all pairs",
        is_leaf=is_type(TaskModelPair),
    ))
    
    ## Create a database record for each ensemble of models trained (i.e. one per disturbance std).   
    # Save the models and training histories to disk.
    model_records = save_all_models(
        db_session,
        NB_ID,
        trained_models,
        train_hps,
        model_hps,
        hps_given_path,
        train_histories,
    )
    
    return model_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    db_session = get_db_session()
    
    default_config = load_default_config(NB_ID)
    
    if args.config is not None:
        config = deep_update(default_config, load_config(args.config))
    else:
        config = default_config
    
    key = jr.PRNGKey(PROJECT_SEED)
    
    model_records = main(config, key)