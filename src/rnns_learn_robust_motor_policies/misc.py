from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import fields
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import re
import subprocess
from types import ModuleType, GeneratorType
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr 
import jax.tree as jt
from jaxtyping import Array, Float
import numpy as np
import pandas as pd
from rich.logging import RichHandler
import yaml

from feedbax.misc import git_commit_id
from feedbax.intervene import AbstractIntervenor, CurlFieldParams, FixedFieldParams
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

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


def camel_to_snake(s: str):
    """Convert camel case to snake case."""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def snake_to_camel(s: str):
    """Convert snake case to camel case."""
    return ''.join(word.title() for word in s.split('_'))


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


def round_to_list(xs: Array, n: int = 5):
    """Rounds floats to a certain number of decimals when casting an array to a list.
    
    This is useful when (e.g.) using `jnp.linspace` to get a sequence of numbers which 
    will be used as keys of a dict, where we want to avoid small floating point variations
    being present in the keys.
    """
    return [round(x, n) for x in xs.tolist()]


def create_arr_df(arr, col_names=None):   
    """Convert a numpy/JAX array into a dataframe of values, with additional columns
    giving the indices of the values in the array.
    
    If the array has complex dtype, split the real and imaginary components
    into separate columns.
    """
    if col_names is None:
        col_names = [f'dim_{i}' for i in range(len(arr.shape))]
    
    # Get all indices including the eigenvalue dimension
    indices = np.indices(arr.shape)
    
    if np.iscomplexobj(arr):
        data_cols = {'real': arr.real.flatten(), 'imag': arr.imag.flatten()}
    else:
        data_cols = {'value': arr.flatten()}
    
    # Create the base dataframe
    df = pd.DataFrame(data_cols)
    
    # Add all dimension indices
    for i, idx_array in enumerate(indices):
        df[col_names[i]] = idx_array.flatten()
    
    return df


def squareform_pdist(xs: Float[Array, "points dims"], ord: int | str | None = 2):
    """Return the pairwise distance matrix between points in `x`.
    
    In the case of `ord=2`, this should be equivalent to:
    
        ```python
        from scipy.spatial.distance import pdist, squareform
        
        squareform(pdist(x, metric='euclidean'))
        ```
    
    However, note that the values for `ord` are those supported
    by `jax.numpy.linalg.norm`. This provides fewer metrics than those 
    supported by `scipy.spatial.distance.pdist`.
    """
    dist = lambda x1, x2: jnp.linalg.norm(x1 - x2, ord=ord)
    row_dist = lambda x: jax.vmap(dist, in_axes=(None, 0))(x, xs)
    return jax.lax.map(row_dist, xs)


def take_model(*args, **kwargs): 
    """Performs `jtree.take` on a feedbax model.
    
    It is currently necessary to use this in place of `jtree.take` when 
    the model contains intervenors with arrays, since those arrays may 
    not have the same batch (e.g. replicate) dimensions as the other 
    model arrays.
    """
    return jtree.filter_wrap(
        lambda x: not is_type(AbstractIntervenor)(x), 
        is_leaf=is_type(AbstractIntervenor),
    )(jtree.take)(
        *args, **kwargs
    )
    
    
def get_dataclass_fields(
    obj: Any, 
    exclude: tuple[str, ...] = (),
    include_internal: bool = False,
) -> dict[str, Any]:
    """Get the fields of a dataclass object as a dictionary."""
    return {
        field.name: getattr(obj, field.name)
        for field in fields(obj)
        if field.name not in exclude
        and (include_internal or not field.metadata.get('internal', False))
    }


def filename_join(strs, joinwith="__"):
    """Helper for formatting filenames from lists of strings."""
    return joinwith.join(s for s in strs if s)


def is_json_serializable(value):
    """Recursive helper function for isinstance-based checking"""
    json_types = (str, int, float, bool, type(None))
    
    if isinstance(value, json_types):
        return True
    elif isinstance(value, Mapping):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in value.items())
    elif isinstance(value, (list, tuple)) and not isinstance(value, GeneratorType):
        return all(is_json_serializable(item) for item in value)
    return False