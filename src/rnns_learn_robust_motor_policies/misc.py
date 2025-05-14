from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import functools
import importlib
import inspect
import json
import logging
from pathlib import Path
import pkgutil
import platform
import re
import subprocess
from types import ModuleType, GeneratorType
import types
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr 
import jax.tree as jt
from jaxtyping import Array, Float, Int
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


def delete_all_files_in_dir(dir_path: Path):
    """Delete all files in a directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")

    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()


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
    if isinstance(x, Mapping):
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


def with_caller_logger(func):
    """
    Decorator that provides the caller's logger to the wrapped function.

    Wrapped functions should accept a `logger: logging.Logger` keyword argument.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If logger is not provided in kwargs, get the caller's logger
        if 'logger' not in kwargs:
            caller_frame = inspect.currentframe().f_back
            caller_module = inspect.getmodule(caller_frame)
            if caller_module:
                kwargs['logger'] = logging.getLogger(caller_module.__name__)
            else:
                kwargs['logger'] = logging.getLogger(func.__module__)
        
        # Call the original function with the resolved logger
        return func(*args, **kwargs)
    
    return wrapper


@with_caller_logger
def get_name_of_callable(
    func: Callable, 
    return_lambda_id: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Returns the name of a callable object, handling different types appropriately.
    
    Args:
        func: The callable object whose name is to be retrieved.
        
    Returns:
        A string representing the callable's name or identifier.
    """
    func_name = getattr(func, '__name__', None)
    
    # Handle lambdas
    if func_name == '<lambda>':
        if return_lambda_id:
            func_id = f"lambda-{str(id(func))}"
            logger.warning(
                f"Generating name for lambda function: returning its id 'lambda-{func_id}'."
            )
            return func_id
        else: 
            logger.warning("Generating name for lambda function: returning 'lambda'.")
            return "lambda"
    
    # Handle partial functions
    elif isinstance(func, functools.partial):
        return get_name_of_callable(func.func)
    
    # Handle method objects (bound or unbound)
    elif inspect.ismethod(func):
        # For bound methods, include class name
        if hasattr(func, '__self__'):
            return f"{func.__self__.__class__.__name__}.{func.__name__}"
        return func.__name__
    
    # Handle callable class instances
    elif callable(func) and not isinstance(func, (types.FunctionType, types.BuiltinFunctionType, type)):
        class_name = func.__class__.__name__
        logger.warning(
            f"Generating name for instance of callable class '{class_name}'. "
            f"Note that instance attributes/state are not captured by this name."
        )
        return class_name
    
    # Regular functions, built-in functions, and classes
    else: 
        if func_name is not None:
            return func.__name__
        else:
            return repr(func)


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


@with_caller_logger
def log_version_info(
    *args: ModuleType, 
    git_modules: Optional[Sequence[ModuleType]] = None,
    python_version: bool = True,
    logger: Optional[logging.Logger] = None,
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


def get_constant_input_fn(x, n_steps: int, n_trials: int):
    return lambda trial_spec, key: (
        jnp.full((n_trials, n_steps - 1), x, dtype=float)
    )


def copy_delattr(obj: Any, *attr_names: str):
    """Return a deep copy of an object, with some attributes removed."""
    obj = deepcopy(obj)
    for attr_name in attr_names:
        delattr(obj, attr_name)
    return obj


def take_non_nan(arr, axis=1):
    # Create tuple of axes to reduce over (all axes except the specified one)
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
    has_nan = jnp.any(jnp.isnan(arr), axis=reduce_axes)
    valid_cols = jnp.where(~has_nan)[0]
    return jnp.take(arr, valid_cols, axis=axis)


def vectors_to_2d_angles(vectors):
    return jnp.arctan2(vectors[..., 1], vectors[..., 0])


def map_fn_over_tree(func, is_leaf: Optional[Callable] = None):
    """Partially applies `jt.map`, for use in functional expressions."""
    @functools.wraps(func)
    def map_fn(tree, *rest):
        return jt.map(func, tree, *rest, is_leaf=is_leaf)
    return map_fn


def normalize(arr, axis=-1):
    return arr / jnp.linalg.norm(arr, axis=axis, keepdims=True)


def ravel_except_last(arr):
    return jnp.reshape(arr, (-1, arr.shape[-1]))


def center_and_rescale(arr, axis=0):
    arr_centered = arr - jnp.nanmean(arr, axis=axis)
    arr_rescaled = arr_centered / jnp.nanmax(arr_centered, axis=axis)
    return arr_rescaled


def _expand_boundary_for_comparison(
    boundary_vals: jnp.ndarray,
    target_ndim: int,
    axis: int,
) -> jnp.ndarray:
    """Expands boundary_vals for broadcasting with target_array's axis_indices."""
    # Assumes boundary_vals.shape matches target_array.shape[:axis_of_comparison]
    expanded = jnp.expand_dims(boundary_vals, axis=axis)
    for i in range(axis + 1, target_ndim):
        expanded = jnp.expand_dims(expanded, axis=i)
    return expanded


def dynamic_slice_with_padding(
    array: Array,
    slice_end_idxs: Int[Array, "..."],
    axis: int,
    slice_start_idxs: Optional[Int[Array, "..."]] = None,
    pad_value: float = jnp.nan,
) -> Array:
    """
    Slices target_array along 'axis' using [start, end) ranges, padding outside.
    slice_params.shape[:-1] should match target_array.shape[:axis].
    """
    if axis < 0:
        axis = array.ndim + axis

    if slice_start_idxs is None:
        slice_start_idxs = jnp.zeros_like(slice_end_idxs)

    axis_indices = jnp.arange(array.shape[axis])
    idx_broadcast_shape = [1] * array.ndim
    idx_broadcast_shape[axis] = array.shape[axis]
    axis_indices_expanded = axis_indices.reshape(idx_broadcast_shape)

    masks = [
        op(
            axis_indices_expanded,
            _expand_boundary_for_comparison(slice_bound, array.ndim, axis),
        )
        for slice_bound, op in zip(
            [slice_start_idxs, slice_end_idxs],
            [jnp.greater_equal, jnp.less],
        )
    ]

    final_mask = jnp.logical_and(*masks)

    return jnp.where(final_mask, array, pad_value)


def get_all_module_names(package_obj, exclude_private: bool = True):
    """Get the names of all modules in a package.
    
    Names include the full package path, e.g. `"some_library.subpackage.module_name"`, 
    even if `package_obj` is `some_library.subpackage`.
    """
    names = []
    if not hasattr(package_obj, '__path__') or not hasattr(package_obj, '__name__'):
        return tuple() # Not a valid package object to inspect
        
    # The prefix ensures names are fully qualified relative to the initial package
    prefix = package_obj.__name__ + '.'
    
    for module_info in pkgutil.walk_packages(package_obj.__path__, prefix):
        is_private = module_info.name.startswith('_') or '._' in module_info.name
        if not module_info.ispkg and not (exclude_private and is_private):
            names.append(module_info.name)
            
    return tuple(names)


def load_module_from_package(name: str, package: ModuleType) -> ModuleType:
    """Given a package object and a string specifying a module within the package, load the module."""
    module_name = f"{package.__name__}.{name}"
    try: 
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error(f"Module '{name}' not found.")
        raise ValueError(f"Module '{name}' not found.")
    return module


def exclude_unshared_keys_and_identical_values(list_of_dicts):
    """Filter dicts in a list to exclude unshared keys, and keys with identical values."""
    if not list_of_dicts:
        return []

    common_keys = set(list_of_dicts[0].keys())
    for d in list_of_dicts[1:]:
        common_keys.intersection_update(d.keys())

    keys_to_exclude = {
        key for key in common_keys
        if all(d[key] == list_of_dicts[0][key] for d in list_of_dicts[1:])
    }

    return [
        {k: v for k, v in original_dict.items() if k not in keys_to_exclude}
        for original_dict in list_of_dicts
    ]