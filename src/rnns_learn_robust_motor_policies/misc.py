from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
from rich.logging import RichHandler
import subprocess
from types import ModuleType
from typing import Optional

import equinox as eqx
import jax.numpy as jnp 
import jax.random as jr 
import jax.tree as jt
from jaxtyping import Array
import yaml

from feedbax.misc import git_commit_id
from feedbax._tree import apply_to_filtered_leaves
from feedbax.intervene import CurlFieldParams, FixedFieldParams

from rnns_learn_robust_motor_policies.tree_utils import subdict


logging.basicConfig(
    format='(%(name)-20s) %(message)s', 
    level=logging.INFO, 
    handlers=[RichHandler(level="NOTSET")],
)
logger = logging.getLogger(__name__)


def dict_str(d, value_format='.2f'):
    """A string representation of a dict that is more filename-friendly than `str` or `repr`."""
    format_string = f"{{k}}-{{v:{value_format}}}"
    return '-'.join(format_string.format(k=k, v=v) for k, v in d.items())


def get_datetime_str():
    return datetime.now().strftime("%Y%m%d-%Hh%M")


def get_gpu_memory(gpu_idx=0):
    """Returns the available memory (in MB) on a GPU. Depends on `nvidia-smi`.
    
    Source: https://stackoverflow.com/a/59571639
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[gpu_idx]


def lohi(x: Iterable):
    """Returns a tuple containing the first and last values of a sequence, mapping, or other iterable."""
    if isinstance(x, dict):
        # TODO: Maybe should return first and last key-value pairs?
        return subdict(x, tuple(lohi(tuple(x.keys()))))
    
    elif isinstance(x, Iterator):
        first = last = next(x)
        for last in x:
            pass
        
    elif isinstance(x, Sequence):
        first = x[0]
        last = x[-1]
    
    elif isinstance(x, Array):
        return lohi(x.tolist())
        
    else: 
        raise ValueError(f"Unsupported type: {type(x)}")
    
    return first, last


def lomidhi(x: Iterable):
    if isinstance(x, dict):
        keys: tuple = tuple(lomidhi(x.keys()))
        return subdict(x, keys)

    elif isinstance(x, Iterator):
        x = tuple(x)
        first, last = lohi(x)
        mid = x[len(x) // 2]
        return first, mid, last

    elif isinstance(x, Array):
        return lomidhi(x.tolist())
    
    else: 
        raise ValueError(f"Unsupported type: {type(x)}")


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_from_json(path):
    with open(path, 'r') as jsonf:
        return json.load(jsonf)
    
    
def write_to_json(tree, file_path):
    arrays, other = eqx.partition(tree, eqx.is_array)
    lists = jt.map(lambda arr: arr.tolist(), arrays)
    serializable = eqx.combine(other, lists)

    with open(file_path, 'w') as jsonf:
        json.dump(serializable, jsonf, indent=4)
        
        
def get_field_amplitude(intervenor_params):
    if isinstance(intervenor_params, FixedFieldParams):
        return jnp.linalg.norm(intervenor_params.field, axis=-1)
    elif isinstance(intervenor_params, CurlFieldParams):
        return jnp.abs(intervenor_params.amplitude)
    else:
        raise ValueError(f"Unknown intervenor parameters type: {type(intervenor_params)}")


def vector_with_gaussian_length(key):
    key1, key2 = jr.split(key)
    
    angle = jr.uniform(key1, (), minval=-jnp.pi, maxval=jnp.pi)
    length = jr.normal(key2, ())

    return length * jnp.array([jnp.cos(angle), jnp.sin(angle)]) 


def log_version_info(
    *args: ModuleType, 
    git_modules: Optional[Sequence[ModuleType]] = None,
    python_version: bool = True,
) -> dict[str, str]:
    version_info: dict[str, str] = {}
    
    if python_version:
        python_ver = platform.python_version()
        version_info["python"] = python_ver
        logger.info(f"python version: {python_ver}")
    
    for package in args:
        version = package.__version__
        version_info[package.__name__] = version
        logger.info(f"{package.__name__} version: {version}")
    
    if git_modules:
        for module in git_modules:
            commit = git_commit_id(module=module)
            version_info[f"{module.__name__} commit"] = commit
            logger.info(f"{module.__name__} commit: {commit}")
    
    return version_info