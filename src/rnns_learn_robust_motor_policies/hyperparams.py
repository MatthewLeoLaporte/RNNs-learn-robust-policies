

from collections.abc import Callable, Sequence
from copy import deepcopy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, TypeVar

import jax.tree as jt
from jaxtyping import ArrayLike, PyTree

from jax_cookbook import is_type, anyf
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.config import load_config, load_default_config
from rnns_learn_robust_motor_policies.constants import LEVEL_LABEL_SEP, get_iterations_to_save_model_parameters
from rnns_learn_robust_motor_policies.tree_utils import (
    tree_level_labels,
    deep_update,
)
from rnns_learn_robust_motor_policies.types import (
    TaskModelPair, 
    LDict,
    TreeNamespace,
    dict_to_namespace,
    is_dict_with_int_keys,
)


# We use LDict labels to identify levels in task-model pair trees
# The label format is expected to be double-underscore separated parts that map to hyperparameter paths
# For example, "train__method" maps to hps.train.method and "train__pert__std" maps to hps.train.pert.std

NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


logger = logging.getLogger(__name__)


def process_hps(hps: TreeNamespace) -> TreeNamespace:
    """Resolve any dependencies and do any clean-up or validation of hyperparameters."""
    # Avoid in-place modification 
    hps = deepcopy(hps)
    
    # Only train configs should have the train key
    if getattr(hps, 'train', None) is not None:
        if getattr(hps.train, 'where', None) is not None:
            # Wrap in an LDict so it doesn't get flattened by `flatten_hps`
            hps.train.where = LDict.of("train__where")(hps.train.where)
        hps.train.intervention_scaleup_batches = [
            hps.train.n_batches_baseline,
            hps.train.n_batches_baseline + hps.train.n_scaleup_batches,
        ]
        hps.train.n_batches = hps.train.n_batches_baseline + hps.train.n_batches_condition
        hps.train.save_model_parameters = get_iterations_to_save_model_parameters(
            hps.train.n_batches
        )
        
    # Not all experiments will load an existing model 
    #? Collapse the load params into the 
    if getattr(hps, 'load', None) is not None:        
        if getattr(hps.load, 'train', None) is not None:
            hps.load.train.where = LDict.of("train__where")(hps.load.train.where)      

    return hps


def load_hps(config_path: str | Path) -> TreeNamespace:
    """Given a path to a YAML hyperparameters file, load and prepare them prior to training.
    
    If the path is not found, pass it as the experiment id to try to get a default config. 
    So you can pass e.g. `"1-1"` to load the default hyperparameters for analysis module 1-1. 
    Note that this is like treating `config_path` as a local path to a YAML file in 
    `rnns_learn_robust_motor_policies.config`.
    """
    try:
        config = load_config(str(config_path))
        expt_id = config['expt_id']
    except FileNotFoundError:
        config = dict()
        expt_id = str(config_path)
    # Load the defaults and update with the user-specified config
    default_config = load_default_config(expt_id)
    config = deep_update(default_config, config)
    # Convert to a (nested) namespace instead of a dict, for attribute access
    hps = dict_to_namespace(config, to_type=TreeNamespace, exclude=is_dict_with_int_keys)
    # Make corrections and add in any derived values
    hps = process_hps(hps)
    return hps


def promote_model_hps(hps: TreeNamespace) -> TreeNamespace:
    """Remove the `model` attribute, and bring its own attributes out to the top level."""
    hps = deepcopy(hps)
    # Bring out the parameters under the `model` key; i.e. "model" won't appear in their flattened keys
    for key in ('model',):
        subtree = getattr(hps, key, None)
        if subtree is not None:
            hps.__dict__.update(subtree.__dict__)
            delattr(hps, key)
    return hps


def flatten_hps(
    hps: TreeNamespace, 
    keep_load: bool = True, 
    is_leaf: Optional[Callable] = anyf(is_type(list), LDict.is_of("train__where")),
    ldict_to_dict: bool = True,
    join_with: str = LEVEL_LABEL_SEP,
) -> TreeNamespace:
    """Flatten the hyperparameter namespace, joining keys with underscores."""
    hps = deepcopy(hps)
    # The structure under the `load` key mimics that of the full hps namespace
    if getattr(hps, 'load', None) is not None:
        if keep_load:
            hps.load = promote_model_hps(hps.load)
        else:
            del hps.load
    
    #! TODO: Don't do this, since we can't unflatten from DB column names later 
    #! without confusing model params with other top-level params
    hps = promote_model_hps(hps)

    hp_values = jt.leaves(hps, is_leaf=is_leaf)
    
    if ldict_to_dict:
        hp_values = [dict(v) if isinstance(v, LDict) else v for v in hp_values]

    return TreeNamespace(**dict(zip(
        jt.leaves(jtree.labels(hps, join_with=join_with, is_leaf=is_leaf)),
        hp_values,
    )))


def update_hps_given_tree_path(hps: TreeNamespace, path: tuple, labels: Sequence[str]) -> TreeNamespace:
    """
    Update hyperparameters based on the path of a task-model pair in the training PyTree.
    
    Args:
        hps: The base hyperparameters
        path: Path to a leaf in the task-model pair tree
        labels: LDict labels for each level in the tree
    
    Returns:
        Updated hyperparameters with values from the path
    """
    hps = deepcopy(hps)
    for node_key, label in zip(path, labels):
        # Split the label to get the path into `hps`
        # For example: "train__method" -> ["train", "method"]
        parts = label.split(LEVEL_LABEL_SEP)

        if not parts:
            continue
            
        # Navigate to the nested attribute and assign
        obj = hps
        for part in parts[:-1]:
            obj = getattr(obj, part)
            
        # Set the final attribute value
        last_part = parts[-1]
        setattr(obj, last_part, node_key.key)

    return hps


def fill_out_hps(hps_common: TreeNamespace, task_model_pairs: PyTree[TaskModelPair, 'T']) -> PyTree[TreeNamespace, 'T']:
    """Given a common set of hyperparameters and a tree of task-model pairs, create a matching tree of 
    pair-specific hyperparameters.

    This works because `task_model_pairs` is a tree of dicts, where each level of the tree is a different 
    dict subtype, and where the keys are the values of hyperparameters. Each dict subtype has a fixed 
    mapping to a particular 
    """
    level_labels = tree_level_labels(task_model_pairs, is_leaf=is_type(TaskModelPair))
    
    # TODO: Use `jt.map_with_path` if updating to new JAX version
    return jt.map(
        lambda _, path: update_hps_given_tree_path(
            hps_common,
            path,
            level_labels,
        ),
        task_model_pairs, 
        jtree.key_tuples(task_model_pairs, is_leaf=is_type(TaskModelPair)),
        is_leaf=is_type(TaskModelPair),
    )


def take_train_histories_hps(hps: TreeNamespace) -> TreeNamespace:
    """Selects specific hyperparameters from a TreeNamespace structure."""
    return TreeNamespace(
        train=TreeNamespace(
            n_batches=hps.train.n_batches,
            batch_size=hps.train.batch_size,
            where=hps.train.where,
            save_model_parameters=hps.train.save_model_parameters,
        ),
        model=TreeNamespace(
            n_replicates=hps.model.n_replicates,
        )
    )


