

from collections.abc import Callable, Sequence
from copy import deepcopy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, TypeVar

import equinox as eqx
from feedbax._tree import is_type
import jax.tree as jt
from feedbax import is_type, tree_key_tuples, tree_labels
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree

from rnns_learn_robust_motor_policies.config import load_config, load_default_config
from rnns_learn_robust_motor_policies.constants import get_iterations_to_save_model_parameters
from rnns_learn_robust_motor_policies.tree_utils import (
    deep_update,
    is_dict_with_int_keys,
    pp,
    tree_level_types,
)
from rnns_learn_robust_motor_policies.types import (
    TaskModelPair, 
    TrainStdDict, 
    TrainingMethodDict,
    TrainWhereDict,
)


# If we construct the pytree of task-model pairs out of these, then in
# `training.train_and_save_models` we can use `fill_out_hps` to automatically
# add to the hyperparameters, key-value pairs where the key corresponds to the 
# dict subtype, and the value corresponds to the key of the respective pair 
# within the dict.
TYPE_HP_KEY_MAPPING = {
    TrainingMethodDict: ("train", "method"),
    # TODO: Rename this to `DisturbanceStdDict`, thus then automate the construction of this mapping?
    # i.e.
    TrainStdDict: ("disturbance", "std"),
}

NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


logger = logging.getLogger(__name__)


@jtu.register_pytree_with_keys_class
class TreeNamespace(SimpleNamespace):
    """A simple namespace that's a PyTree.

    This is useful when we want to attribute-like access to the data in
    a nested dict. For example, `hyperparameters['train']['n_batches']` 
    becomes `TreeNamespace(**hyperparameters).train.n_batches`.
    """
    def tree_flatten_with_keys(self):
        children_with_keys = [(jtu.GetAttrKey(k), v) for k, v in self.__dict__.items()]
        aux_data = self.__dict__.keys()
        return children_with_keys, aux_data      

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))

    def update_none_leaves(self, other):
        # I would just use `jt.map` or `eqx.combine` to do this, however I don't want to assume
        # that `other` will have identical PyTree structure to `self` -- only that it contains at 
        # least the keys whose values are `None` in `self`.
        #? Could work on flattened trees.
        def _update_none_leaves(target: TreeNamespace, source: TreeNamespace) -> TreeNamespace:
            result = deepcopy(target)
            source = deepcopy(source)

            for attr_name in vars(result):
                result_value = getattr(result, attr_name)
                source_value = getattr(source, attr_name, None)

                if result_value is None:
                    if source_value is None:
                        raise ValueError(f"Cannot replace `None` value of key {attr_name}; no matching key available in source")
                    setattr(result, attr_name, source_value)

                elif isinstance(result_value, TreeNamespace):
                    if source_value is None:
                        raise ValueError(f"")
                    if not isinstance(source_value, TreeNamespace):
                        raise ValueError(f"Source must contain all the parent keys (but not necessarily all the leaves) of the target")
                    setattr(result, attr_name, _update_none_leaves(result_value, source_value))

            return result
        return _update_none_leaves(self, other)

    def __or__(self, other: 'TreeNamespace | dict') -> 'TreeNamespace':
        """Merge two TreeNamespaces, with values from other taking precedence.
        
        Handles nested TreeNamespaces recursively.
        """
        result = deepcopy(self)
        
        if isinstance(other, dict):
            other = dict_to_namespace(other, to_type=TreeNamespace, exclude=is_dict_with_int_keys)

        for attr_name, other_value in vars(other).items():
            self_value = getattr(result, attr_name, None)
            
            if isinstance(other_value, TreeNamespace) and isinstance(self_value, TreeNamespace):
                # Recursively merge nested TreeNamespaces
                setattr(result, attr_name, self_value | other_value)
            else:
                # Simply update, when at least one side isn't a TreeNamespace
                setattr(result, attr_name, other_value)
                
        return result


def process_hps(hps: TreeNamespace) -> TreeNamespace:
    """Resolve any dependencies and do any clean-up or validation of hyperparameters."""
    # Avoid in-place modification 
    hps = deepcopy(hps)
    
    # Only train configs should have the train key
    if getattr(hps, 'train', None) is not None:
        if getattr(hps.train, 'where', None) is not None:
            hps.train.where = TrainWhereDict(hps.train.where)
        hps.train.intervention_scaleup_batches = [
            hps.train.n_batches_baseline,
            hps.train.n_batches_baseline + hps.train.n_scaleup_batches,
        ]
        hps.train.n_batches = hps.train.n_batches_baseline + hps.train.n_batches_condition
        hps.train.save_model_parameters = get_iterations_to_save_model_parameters(
            hps.train.n_batches
        )
        
    # Not all experiments will load an existing model 
    if getattr(hps, 'load', None) is not None:        
        if getattr(hps.load, 'train', None) is not None:
            hps.load.train.where = TrainWhereDict(hps.load.train.where)

    return hps


def load_hps(config_path: str | Path) -> TreeNamespace:
    """Given a path to a YAML hyperparameters file, load and prepare them prior to training."""
    config = load_config(str(config_path))
    # Load the defaults and update with the user-specified config
    default_config = load_default_config(config['expt_id'])
    config = deep_update(default_config, config)
    # 1) Convert to a (nested) namespace instead of a dict,
    #    so we can refer to keys as attributes
    # 2) Make corrections and add in any derived values
    hps = dict_to_namespace(config, to_type=TreeNamespace, exclude=is_dict_with_int_keys)
    hps = process_hps(hps)
    return hps


def promote_model_hps(hps: TreeNamespace) -> TreeNamespace:
    """Remove the `model` attribute, and bring its own attributes out to the top level."""
    hps = deepcopy(hps)
    # Bring out the parameters under the `model` key; i.e. "model" won't appear in their flattened keys
    for key in ('model',):
        if getattr(hps, key, None) is not None:
            hps.__dict__.update(getattr(hps, key).__dict__)
            delattr(hps, key)
    return hps


def flatten_hps(
    hps: TreeNamespace, 
    keep_load: bool = True, 
    is_leaf: Optional[Callable] = is_type(list, TrainWhereDict),
) -> TreeNamespace:
    """Flatten the hyperparameter namespace, joining keys with underscores."""
    hps = deepcopy(hps)
    # The structure under the `load` key mimics that of the full hps namespace
    if getattr(hps, 'load', None) is not None:
        if keep_load:
            hps.load = promote_model_hps(hps.load)
        else:
            del hps.load

    hps = promote_model_hps(hps)

    return TreeNamespace(**dict(zip(
        jt.leaves(tree_labels(hps, join_with='_', is_leaf=is_leaf)),
        jt.leaves(hps, is_leaf=is_leaf),
    )))


def update_hps_given_tree_path(hps: TreeNamespace, path: tuple, types: Sequence) -> TreeNamespace:
    """Given the path of a task-model pair in the training PyTree, """
    hps = deepcopy(hps)
    for node_key, type_ in zip(path, types):
        hps_key, hps_subkey = TYPE_HP_KEY_MAPPING[type_]
        setattr(getattr(hps, hps_key), hps_subkey, node_key.key)
    return hps


def fill_out_hps(hps_common: TreeNamespace, task_model_pairs: PyTree[TaskModelPair, 'T']) -> PyTree[TreeNamespace, 'T']:
    """Given a common set of hyperparameters and a tree of task-model pairs, create a matching tree of 
    pair-specific hyperparameters.

    This works because `task_model_pairs` is a tree of dicts, where each level of the tree is a different 
    dict subtype, and where the keys are the values of hyperparameters. Each dict subtype has a fixed 
    mapping to a particular 
    """
    level_types = tree_level_types(task_model_pairs)
    return jt.map(
        lambda _, path: update_hps_given_tree_path(
            hps_common,
            path,
            level_types,
        ),
        task_model_pairs, tree_key_tuples(task_model_pairs, is_leaf=is_type(TaskModelPair)),
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


def _convert_value(value: Any, to_type: type, from_type: type, exclude: Callable) -> Any:
    recurse_func = lambda x: _convert_value(x, to_type, from_type, exclude)
    map_recurse_func = lambda tree: jt.map(recurse_func, tree, is_leaf=is_type(from_type))

    if exclude(value):
        subtrees, treedef = eqx.tree_flatten_one_level(value)
        subtrees = [map_recurse_func(subtree) for subtree in subtrees]
        return jt.unflatten(treedef, subtrees)
    
    elif isinstance(value, from_type):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dict or namespace, got {type(value)}")
        
        return to_type(**{
            str(k): recurse_func(v) 
            for k, v in value.items()
        })
    
    elif isinstance(value, (str, ArrayLike, type(None))):
        return value
    
    # Map over any remaining PyTrees, except 
    elif isinstance(value, PyTree):
        # `object` is an atomic PyTree, so without this check we'll get infinite recursion
        if value is not object:
            return map_recurse_func(value)
    
    return value


def convert_kwargy_node_type(x, to_type: type, from_type: type, exclude: Callable = lambda x: False):
    """Convert a nested dictionary to a nested SimpleNamespace.

    !!! dev 
        This should convert all the dicts to namespaces, even if the dicts are not contiguous all 
        the way down (e.g. a dict in a list in a list in a dict)
    """
    return _convert_value(x, to_type, from_type, exclude)


def namespace_to_dict(
    ns: SimpleNamespace,
    to_type: type[DT] = dict,
    exclude: Callable = lambda x: False,
) -> DT:
    """Convert a nested SimpleNamespace to a nested dictionary.

    This is the inverse operation of dict_to_namespace.
    """
    return convert_kwargy_node_type(ns, to_type=to_type, from_type=SimpleNamespace, exclude=exclude)


def dict_to_namespace(
    d: dict,
    to_type: type[NT] = SimpleNamespace,
    exclude: Callable = lambda x: False,
) -> NT:
    """Convert a nested dictionary to a nested SimpleNamespace.

    This is the inverse operation of namespace_to_dict.
    """
    return convert_kwargy_node_type(d, to_type=to_type, from_type=dict, exclude=exclude)