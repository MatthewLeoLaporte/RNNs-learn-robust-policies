#!/usr/bin/env python
""""""

from functools import reduce
from importlib import resources
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
from copy import deepcopy
from itertools import product
import logging
from typing import Dict, List, Any
import warnings
import yaml

import equinox as eqx 
import jax 
import jax.random as jr
import optax

import feedbax

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.database import get_db_session
from rnns_learn_robust_motor_policies.hyperparams import load_hps
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training.train import train_and_save_models
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace


# TODO: Figure out why the warning from this module appears.
# It seems to have to do with `_train_step` in `feedbax.train`
warnings.filterwarnings("ignore", module="equinox._module")


logger = logging.getLogger(os.path.basename(__file__))


CONFIG_FILENAME = 'train_all'
CONFIGS = resources.files('rnns_learn_robust_motor_policies.config')


def unwrap_value(v: Any) -> Any:
    """Unwrap a value if it's wrapped in a 'value' key."""
    return v['value'] if isinstance(v, dict) and 'value' in v else v


def process_section(section_params: Dict) -> List[Dict]:
    """Process a single section's parameters into a list of configs."""
    # Find parameters that are lists (but not wrapped in 'value')
    list_params = {
        k: v for k, v in section_params.items() 
        if isinstance(v, list) and not isinstance(v, dict)
    }
    
    if not list_params:
        return [{k: unwrap_value(v) for k, v in section_params.items()}]
        
    # Verify all lists have the same length
    lengths = set(len(v) for v in list_params.values())
    if len(lengths) > 1:
        raise ValueError("All parameter lists must have same length")
    
    # Create a config for each index
    return [
        {
            k: v[i] if k in list_params else unwrap_value(v)
            for k, v in section_params.items()
        }
        for i in range(next(iter(lengths)))
    ]


def merge_dicts_nested(d1: Dict, d2: Dict) -> Dict:
    """Merge two dictionaries, combining nested dicts."""
    result = deepcopy(d1)
    for k, v in d2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts_nested(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def parse_batch_config(config: Dict) -> Dict[int, List[Dict]]:
    """Parse a batch configuration into a list of individual configs."""
    
    def process_node(node: Dict) -> List[Dict]:
        node = {k: v for k, v in node.items() if k != 'name'}
        
        if 'cases' in node and 'product' in node:
            raise ValueError("Node cannot have both 'cases' and 'product' keys")
        
        if 'cases' in node:
            return [cfg for case in node['cases'] for cfg in process_node(case)]
            
        if 'product' in node:
            factor_configs = [process_node(factor) for factor in node['product']]
            return [
                reduce(merge_dicts_nested, configs)
                for configs in product(*factor_configs)
            ]
        
        # Base case: process each section
        section_configs = {
            section: process_section(params)
            for section, params in node.items()
        }
        
        return [
            {
                section: cfg
                for section, cfg in zip(section_configs, configs)
            }
            for configs in product(*(section_configs.values()))
        ]

    return {
        expt_id: process_node(expt_config) 
        for expt_id, expt_config in config.items()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train all project models, based on the batch config.")
    # Because of the behaviour of `load_hps`, config_path can also be the `expt_id: str` (i.e. YAML 
    # filename relative to `../src/rnns_learn_robust_motor_policies/config`) of a default config to load. 
    # This assumes there is no file whose relative path is identical to that `expt_id`.
    parser.add_argument("--config-path", type=str, default="", help="Path to the config file.")
    parser.add_argument("--expt-id", type=str, default="", help="Process only this experiment, instead of all of them.")
    parser.add_argument("--untrained-only", action='store_false', help="Only train models which appear not to have been trained yet.")
    parser.add_argument("--postprocess", action='store_false', help="Postprocess each model after training.")
    parser.add_argument("--n-std-exclude", type=int, default=2, help="In postprocessing, exclude model replicates with n_std greater than this value.")
    parser.add_argument("--save-figures", action='store_true', help="Save figures in postprocessing.")
    args = parser.parse_args()
    
    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    db_session = get_db_session()
    
    key = jr.PRNGKey(PROJECT_SEED)
    
    if not args.config_path:
        resource = CONFIGS.joinpath(f'{CONFIG_FILENAME}.yml')
        batch_config = yaml.safe_load(resource.open('r'))
    else: 
        with open(args.config_path, 'r') as f:
            batch_config = yaml.safe_load(f)
            
    configs = parse_batch_config(batch_config)
    
    if not args.expt_id == "":
        configs = {args.expt_id: configs[args.expt_id]}      
    
    logger.info(f"Training models for experiments: {', '.join(str(s) for s in configs.keys())}")  
    
    # Iterate over each experiment
    
    for expt_id, expt_configs in configs.items():
    
        # Get defaults for this experiment
        hps_common: TreeNamespace = load_hps(str(expt_id))
        
        # Train each set of models with its respective config
        for i, config in enumerate(expt_configs):
            
            logger.info(f"Training models for experiment {expt_id}, config {i} of {len(expt_configs)}")
            
            hps = hps_common | config
            
            trained_models, train_histories, model_records = train_and_save_models(
                db_session, 
                hps,
                key,
                untrained_only=args.untrained_only,
                postprocess=args.postprocess,
                n_std_exclude=args.n_std_exclude,
                save_figures=args.save_figures,
                version_info=version_info,
            )