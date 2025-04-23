#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os


os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
import logging
from pathlib import Path

# NOTE: JAX arrays are not directly picklable if they contain device memory references.
# Since we're using pickle to cache states which may contain JAX arrays, we rely on JAX's
# implicit handling of arrays during pickling (it should work for CPU arrays and most
# host-accessible device arrays).
import jax.random as jr

from rnns_learn_robust_motor_policies.analysis.execution import run_analysis_module
from rnns_learn_robust_motor_policies.config import PATHS, PRNG_CONFIG
from rnns_learn_robust_motor_policies.database import (
    check_model_files,
    get_db_session, 
)
from rnns_learn_robust_motor_policies.hyperparams import load_hps


logger = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    # Because of the behaviour of `load_hps`, config_path can also be the `expt_id: str` (i.e. YAML 
    # filename relative to `../src/rnns_learn_robust_motor_policies/config`) of a default config to load. 
    # This assumes there is no file whose relative path is identical to that `expt_id`.
    parser.add_argument("config_path", type=str, help="Path to the config file, or `expt_id` of a default config.")
    parser.add_argument("--fig-dump-path", type=str, help="Path to dump figures.")
    parser.add_argument("--fig-dump-formats", type=str, default="html,webp,svg", 
                      help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')")
    parser.add_argument("--no-pickle", action="store_true", help="Do not use pickle for states (don't load existing or save new).")
    parser.add_argument("--retain-past-fig-dumps", action="store_true", help="Do not save states to pickle.")
    parser.add_argument("--states-pkl-dir", type=str, default=None, help="Alternative directory for state pickle files (default: PATHS.states_tmp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the analysis.")
    
    args = parser.parse_args()
    
    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)
    
    # Parse the figure dump formats
    fig_dump_formats = args.fig_dump_formats.split(',')
    
    # Set states pickle directory
    states_pkl_dir = Path(args.states_pkl_dir) if args.states_pkl_dir else PATHS.states_tmp
    
    _ = run_analysis_module(
        config_path=args.config_path, 
        fig_dump_path=args.fig_dump_path, 
        fig_dump_formats=fig_dump_formats, 
        retain_past_fig_dumps=args.retain_past_fig_dumps,
        no_pickle=args.no_pickle,
        states_pkl_dir=states_pkl_dir,
        key=key
    )

    #! TMP: Keep the debugger alive.
    raise RuntimeError("Stop here.")
    
    
    