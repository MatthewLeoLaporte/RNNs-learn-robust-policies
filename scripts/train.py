#!/usr/bin/env python
"""From the command line, train some models according to some configuration. 

Load the config and pass it to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os

import equinox as eqx
import jax
import optax

import feedbax

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies.misc import log_version_info


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
import warnings
from rnns_learn_robust_motor_policies._warnings import enable_warning_dedup

import jax.random as jr

from rnns_learn_robust_motor_policies.config import PRNG_CONFIG
from rnns_learn_robust_motor_policies.training import train_and_save_models
from rnns_learn_robust_motor_policies.config import load_hps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("expt_name", type=str, help="Name of the training experiment to run.")  
    parser.add_argument("--untrained-only", action='store_false', help="Only train models which appear not to have been trained yet.")
    parser.add_argument("--postprocess", action='store_false', help="Postprocess each model after training.")
    parser.add_argument("--n-std-exclude", type=int, default=2, help="In postprocessing, exclude model replicates with n_std greater than this value.")
    parser.add_argument("--save-figures", action='store_true', help="Save figures in postprocessing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the training.")
    parser.add_argument("--show-duplicate-warnings", action="store_true",
                        help="If set, all occurrences of each distinct warning message are shown.")
    
    args = parser.parse_args()
    
    # Optionally install warning de-duplication.
    if not args.show_duplicate_warnings:
        enable_warning_dedup()

    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)

    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    hps = load_hps(args.expt_name, config_type='training')
    
    trained_models, train_histories, model_records = train_and_save_models(
        hps=hps,
        expt_name=args.expt_name,
        untrained_only=args.untrained_only,
        postprocess=args.postprocess,
        n_std_exclude=args.n_std_exclude,
        save_figures=args.save_figures,
        version_info=version_info,
        key=key,
    )