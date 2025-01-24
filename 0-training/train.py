"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

#!/usr/bin/env python
# coding: utf-8

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
import warnings

import equinox as eqx
import jax
import jax.random as jr
import optax 

import feedbax

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED

from rnns_learn_robust_motor_policies.database import (
    get_db_session,
)
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training import (
    train_and_save_models,
)


# TODO: Figure out why the warning from this module appears.
# It seems to have to do with `_train_step` in `feedbax.train`
warnings.filterwarnings("ignore", module="equinox._module")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument("--untrained-only", action='store_false', help="Only train models which appear not to have been trained yet.")
    parser.add_argument("--postprocess", action='store_false', help="Postprocess each model after training.")
    parser.add_argument("--n-std-exclude", type=int, default=2, help="In postprocessing, exclude model replicates with n_std greater than this value.")
    parser.add_argument("--save-figures", action='store_false', help="Save figures in postprocessing.")
    args = parser.parse_args()
    
    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    db_session = get_db_session()
    
    key = jr.PRNGKey(PROJECT_SEED)
    
    trained_models, train_histories, model_records = train_and_save_models(
        db_session, 
        args.config_path, 
        key,
        untrained_only=args.untrained_only,
        postprocess=args.postprocess,
        n_std_exclude=args.n_std_exclude,
        save_figures=args.save_figures,
    )