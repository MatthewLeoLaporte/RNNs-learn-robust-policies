#!/usr/bin/env python
# coding: utf-8

from collections.abc import Callable
import os
from pathlib import Path
from typing import Optional

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
from rnns_learn_robust_motor_policies.tree_utils import (
    deep_update,
)
from rnns_learn_robust_motor_policies.types import TaskModelPair
from rnns_learn_robust_motor_policies.train_setup import (
    train_pair,
    train_setup,
)
from rnns_learn_robust_motor_policies.train_setup_part1 import (
    custom_hps_given_path as custom_hps_given_path_1,
    get_train_pairs as get_train_pairs_1,
)
from rnns_learn_robust_motor_policies.train_setup_part2 import (
    custom_hps_given_path as custom_hps_given_path_2,
    get_train_pairs as get_train_pairs_2,
)
from rnns_learn_robust_motor_policies.types import TrainStdDict


# These are the different types of training run, i.e. respective to parts/phases of the study.
# TODO: Eliminate `custom_hps_given_path` completely
VARIANTS = {
    1: (get_train_pairs_1, custom_hps_given_path_1), 
    2: (get_train_pairs_2, custom_hps_given_path_2),   
}
# The training logic is identical except importantly for the function `get_train_pairs`, which 
# determines the structure of the PyTree of task-model tuples, which of course may vary between 
# training experiments.


def load_hps(config_path: str | Path) -> dict:
    """Given a path to a YAML config..."""
    config = load_config(config_path)
    # Load the defaults and update with the user-specified config
    default_config = load_default_config(config['id'])
    config = deep_update(default_config, config)
    # Make corrections and add in any derived values.
    hps = process_hps(config)  
    return hps


def train_and_save_models(
    db_session,
    config_path: str | Path, 
    key: PRNGKeyArray,
):
    """Given a path to a YAML config, execute the respective training run.
    
    The config must have a top-level key `id` whose positive integer value 
    indicates which training experiment to run. 
    """
    key_init, key_train, key_eval = jr.split(key, 3)
    
    hps = load_hps(config_path)
    
    # from rnns_learn_robust_motor_policies.tree_utils import pp
    # pp(hps)
    # raise RuntimeError
    
    # User specifies which variant to run using the `id` key
    get_train_pairs, custom_hps_given_path = VARIANTS[hps['id']]
    
    task_model_pairs = get_train_pairs(hps, key_init)
    
    trainer, loss_func = train_setup(hps['train'])
    
    # Convert string representations of where-functions to actual functions.
    # 
    #   - Strings are easy to serialize, or to specify in config files; functions are not.
    #   - These where-functions are for selecting the trainable nodes in the pytree of model 
    #     parameters.
    #
    where_train = {
        i: attr_str_tree_to_where_func(strs) 
        for i, strs in hps['train']['where_train_strs'].items()
    }
    
    ## Train all the models.
    # Organize the constant arguments for the calls to `train_pair`
    train_args = dict(
        ensembled=True,
        loss_func=loss_func,
        # TODO: Is this correct? Or should we pass the task for the respective training method?
        task_baseline=jt.leaves(task_model_pairs, is_leaf=is_type(TrainStdDict))[0][0].task, 
        where_train=where_train,
        batch_size=hps['train']['batch_size'], 
        log_step=500,
        save_model_parameters=hps['train']['save_model_parameters'],
        state_reset_iterations=hps['train']['state_reset_iterations'],
        # disable_tqdm=True,
    )

    # The imported `train_pair` function actually runs the trainer
    trained_models, train_histories = tree_unzip(tree_map_tqdm(
        #! Use the same PRNG key for all training runs
        partial(train_pair, trainer, hps['train']['n_batches'], key=key_train, **train_args),
        task_model_pairs,
        label="Training all pairs",
        is_leaf=is_type(TaskModelPair),
    ))
    
    ## Create a database record for each ensemble of models trained (i.e. one per disturbance std).   
    # Save the models and training histories to disk.
    model_records = save_all_models(
        db_session,
        hps['id'],
        trained_models,
        hps,
        custom_hps_given_path,
        train_histories,
    )
    
    return model_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    
    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    db_session = get_db_session()
    
    key = jr.PRNGKey(PROJECT_SEED)
    
    model_records = train_and_save_models(
        db_session, 
        args.config_path, 
        key,
    )