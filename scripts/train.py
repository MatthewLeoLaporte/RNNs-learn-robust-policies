#!/usr/bin/env python
"""From the command line, train some models according to some configuration. 

Load the config and pass it to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

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
from rnns_learn_robust_motor_policies.config import PRNG_CONFIG
from rnns_learn_robust_motor_policies.database import get_db_session
from rnns_learn_robust_motor_policies.hyperparams import load_hps
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training import train_and_save_models
from rnns_learn_robust_motor_policies.types import TreeNamespace


# TODO: Figure out why the warning from this module appears.
# It seems to have to do with `_train_step` in `feedbax.train`
warnings.filterwarnings("ignore", module="equinox._module")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    # Because of the behaviour of `load_hps`, config_path can also be the `expt_id: str` (i.e. YAML 
    # filename relative to `../src/rnns_learn_robust_motor_policies/config`) of a default config to load. 
    # This assumes there is no file whose relative path is identical to that `expt_id`.
    parser.add_argument("config_path", type=str, help="Path to the config file.")  
    parser.add_argument("--untrained-only", action='store_false', help="Only train models which appear not to have been trained yet.")
    parser.add_argument("--postprocess", action='store_false', help="Postprocess each model after training.")
    parser.add_argument("--n-std-exclude", type=int, default=2, help="In postprocessing, exclude model replicates with n_std greater than this value.")
    parser.add_argument("--save-figures", action='store_true', help="Save figures in postprocessing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the training.")
    
    args = parser.parse_args()
    
    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)
    
    trained_models, train_histories, model_records = train_and_save_models(
        config_path=args.config_path,
        untrained_only=args.untrained_only,
        postprocess=args.postprocess,
        n_std_exclude=args.n_std_exclude,
        save_figures=args.save_figures,
        key=key,
    )