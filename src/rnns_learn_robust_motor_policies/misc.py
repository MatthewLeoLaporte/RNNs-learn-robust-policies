from collections.abc import Iterable, Iterator, Mapping
import json
import subprocess

import equinox as eqx
import jax.tree as jt

from feedbax._tree import apply_to_filtered_leaves


def dict_str(d, value_format='.2f'):
    """A string representation of a dict that is more filename-friendly than `str` or `repr`."""
    format_string = f"{{k}}-{{v:{value_format}}}"
    return '-'.join(format_string.format(k=k, v=v) for k, v in d.items())


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
        return lohi(tuple(x.values()))
    
    elif isinstance(x, Iterator):
        first = last = next(x)
        for last in iterable:
            pass
        
    else:
        first = x[0]
        last = x[-1]
    
    return first, last


def load_from_json(path):
    with open(path, 'r') as jsonf:
        return json.load(jsonf)
    
    
def write_to_json(tree, file_path):
    arrays, other = eqx.partition(tree, eqx.is_array)
    lists = jt.map(lambda arr: arr.tolist(), arrays)
    serializable = eqx.combine(other, lists)

    with open(file_path, 'w') as jsonf:
        json.dump(serializable, jsonf, indent=4)