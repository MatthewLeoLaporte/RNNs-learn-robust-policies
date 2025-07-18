#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


import argparse
import logging
from pathlib import Path
import warnings
from rlrmp._warnings import enable_warning_dedup

# NOTE: JAX arrays are not directly picklable if they contain device memory references.
# Since we're using pickle to cache states which may contain JAX arrays, we rely on JAX's
# implicit handling of arrays during pickling (it should work for CPU arrays and most
# host-accessible device arrays).
import jax.random as jr

from rlrmp.analysis.execution import run_analysis_module
from rlrmp.config import PATHS, PRNG_CONFIG


logger = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("analysis_name", type=str, help="Name of the analysis module to run; e.g. part1.plant_perts")
    parser.add_argument("--fig-dump-path", type=str, help="Path to dump figures.")
    parser.add_argument("--fig-dump-formats", type=str, default="html,webp,svg", 
                      help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')")
    parser.add_argument("--no-pickle", action="store_true", help="Do not use pickle for states (don't load existing or save new).")
    parser.add_argument("--retain-past-fig-dumps", action="store_true", help="Do not save states to pickle.")
    parser.add_argument("--states-pkl-dir", type=str, default=None, help="Alternative directory for state pickle files (default: PATHS.cache/'states')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the analysis.")
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
    
    # Parse the figure dump formats
    fig_dump_formats = args.fig_dump_formats.split(',')
    
    # Set states pickle directory
    states_pkl_dir = Path(args.states_pkl_dir) if args.states_pkl_dir else PATHS.cache / "states"
    
    _ = run_analysis_module(
        analysis_name=args.analysis_name, 
        fig_dump_path=args.fig_dump_path, 
        fig_dump_formats=fig_dump_formats, 
        retain_past_fig_dumps=args.retain_past_fig_dumps,
        no_pickle=args.no_pickle,
        states_pkl_dir=states_pkl_dir,
        key=key
    )
    
    
    