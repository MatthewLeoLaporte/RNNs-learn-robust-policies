from importlib import resources
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Optional, TypeVar
import yaml

import jax.tree as jt

from rnns_learn_robust_motor_policies.types import TreeNamespace, dict_to_namespace


logger = logging.getLogger(__name__)


CONFIG_DIR_ENV_VAR_NAME = 'RLRMP_CONFIG_DIR'

def setup_paths(paths_ns: TreeNamespace):
    base_path = Path(paths_ns.base)

    def _setup_path(path_str: str):
        if path_str == 'base':
            return base_path
        else:
            path = base_path / path_str
            path.mkdir(parents=True, exist_ok=True)
            return path

    return jt.map(_setup_path, paths_ns)


T = TypeVar('T', bound=SimpleNamespace)


def get_user_config_dir():
    """Get user config directory from environment variable, or return None."""
    env_config_dir = os.environ.get(CONFIG_DIR_ENV_VAR_NAME)
    if env_config_dir is None:
        return 
    else:
        return Path(env_config_dir).expanduser() 


def load_config(name: str, config_type: Optional[Literal['training', 'analysis']] = None):
    """Load the contents of a project YAML config file resource as a nested dict."""
    name_parts = name.split('.')
    config_name = name_parts[-1]
    
    # If the user has specified a config directory, try to load the paths config from it
    user_config_dir = get_user_config_dir()
    if user_config_dir is not None:
        subpath = '/'.join(name_parts[:-1])
        try:
            with open(user_config_dir / subpath / f'{name}.yml') as f:
                return yaml.safe_load(f)
        except:  
            logger.info(f'Config file {f"{subpath}/{config_name}.yml"} not found in user config directory; using default.')
    
    if config_type is None:
        subpackage_name = 'rnns_learn_robust_motor_policies.config'
    else:
        subpackage_name = f'rnns_learn_robust_motor_policies.config.{config_type}'
    
    subpackage_name = '.'.join([subpackage_name, *name_parts[:-1]])
    
    # Otherwise, load the default
    with resources.open_text(subpackage_name, f'{config_name}.yml') as f:
        return yaml.safe_load(f)


def load_config_as_ns(
    name: str, 
    config_type: Optional[Literal['training', 'analysis']] = None,
    to_type: type[T] = TreeNamespace,
) -> T:
    """Load the contents of a project YAML config file resource as a namespace."""
    return dict_to_namespace(load_config(name, config_type), to_type=to_type)


# Load project-wide configuration from YAML resources in the `config` subpackage
CONSTANTS: TreeNamespace = load_config_as_ns("constants")
LOGGING_CONFIG: TreeNamespace = load_config_as_ns("logging")
PATHS: TreeNamespace = setup_paths(load_config_as_ns("paths"))
PLOTLY_CONFIG: TreeNamespace = load_config_as_ns("plotly")
PRNG_CONFIG: TreeNamespace = load_config_as_ns("prng")
STRINGS: TreeNamespace = load_config_as_ns("strings")