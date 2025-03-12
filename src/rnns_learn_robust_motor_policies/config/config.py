from importlib import resources
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Optional, TypeVar
import yaml

import jax.tree as jt

from rnns_learn_robust_motor_policies.types import TreeNamespace, dict_to_namespace


CONFIG_DIR_ENV_VAR_NAME = 'RLRMP_CONFIG_DIR'


def setup_path(path_str: str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


T = TypeVar('T', bound=SimpleNamespace)


def get_user_config_dir():
    """Get user config directory from environment variable or return None"""
    env_config_dir = os.environ.get(CONFIG_DIR_ENV_VAR_NAME)
    if env_config_dir is None:
        return 
    else:
        return Path(env_config_dir).expanduser() 


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def load_yaml_config(name: str, config_type: Optional[Literal['training', 'analysis']] = None):
    """Load the contents of a project YAML config file resource as a nested dict."""
    user_config_dir = get_user_config_dir()
    
    # If the user has specified a config directory, try to load the paths config from it
    if user_config_dir is not None:
        try:
            with open(user_config_dir / f'{name}.yml') as f:
                return yaml.safe_load(f)
        except:  # TODO
            pass
    
    if config_type is None:
        subpackage_name = 'rnns_learn_robust_motor_policies.config'
    else:
        subpackage_name = f'rnns_learn_robust_motor_policies.config.{config_type}'
    
    # Otherwise, load the default
    with resources.open_text(subpackage_name, f'{name}.yml') as f:
        return yaml.safe_load(f)


def load_yaml_config_as_ns(
    name: str, 
    config_type: Optional[Literal['training', 'analysis']] = None,
    to_type: type[T] = TreeNamespace,
) -> T:
    """Load the contents of a project YAML config file resource as a namespace."""
    return dict_to_namespace(load_yaml_config(name, config_type), to_type=to_type)


# Load project-wide configuration from YAML resources in the `config` subpackage
CONSTANTS: TreeNamespace = load_yaml_config_as_ns("constants")
LOGGING_CONFIG: TreeNamespace = load_yaml_config_as_ns("logging")
PATHS: TreeNamespace = jt.map(
    setup_path,
    load_yaml_config_as_ns("paths"),
)
PLOTLY_CONFIG: TreeNamespace = load_yaml_config_as_ns("plotly")
PRNG_CONFIG: TreeNamespace = load_yaml_config_as_ns("prng")
STRINGS: TreeNamespace = load_yaml_config_as_ns("strings")