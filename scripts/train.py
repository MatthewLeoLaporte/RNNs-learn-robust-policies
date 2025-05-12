#!/usr/bin/env python
"""From the command line, train some models according to some configuration. 

Load the config and pass it to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
import warnings

import jax.random as jr

from rnns_learn_robust_motor_policies.config import PRNG_CONFIG
from rnns_learn_robust_motor_policies.training import train_and_save_models


# TODO: Figure out why the warning from this module appears.
# It seems to have to do with `_train_step` in `feedbax.train`
warnings.filterwarnings("ignore", module="equinox._module")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("expt_name", type=str, help="Name of the training experiment to run.")  
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
        expt_name=args.expt_name,
        untrained_only=args.untrained_only,
        postprocess=args.postprocess,
        n_std_exclude=args.n_std_exclude,
        save_figures=args.save_figures,
        key=key,
    )