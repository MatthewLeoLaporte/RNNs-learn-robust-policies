from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
import time
from typing import Any, Literal, Optional
import fnmatch
import json 
import os

import equinox as eqx
from ipyfilechooser import FileChooser
from ipywidgets import HTML
from IPython.display import display
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray

from feedbax import (
    is_type, 
    is_module, 
    load, 
    tree_set_scalar,
)
from feedbax.loss import AbstractLoss
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.noise import Multiplicative, Normal
from feedbax.task import SimpleReaches
from feedbax.train import (
    TaskTrainer,
    TaskTrainerHistory, 
    init_task_trainer_history,
)
from feedbax._tree import (
    tree_zip_named, 
    tree_unzip,
)
from feedbax.xabdeef.losses import simple_reach_loss
from jaxtyping import PyTree
import optax

from rnns_learn_robust_motor_policies import MODELS_DIR
from rnns_learn_robust_motor_policies.constants import (
    TASK_EVAL_PARAMS,
    N_STEPS,
    WORKSPACE,
    get_iterations_to_save_model_parameters,
)
from rnns_learn_robust_motor_policies.database import (
    get_model_record,
    save_model_and_add_record,
)
from rnns_learn_robust_motor_policies.loss import get_readout_norm_loss
from rnns_learn_robust_motor_policies.misc import (
    take_model,
)
from rnns_learn_robust_motor_policies.tree_utils import (
    dictmerge, 
    index_multi,
    map_kwargs_to_dict,
    subdict,
)
from rnns_learn_robust_motor_policies.types import (
    TrainStdDict,
    TrainingMethodDict,
)


def get_base_task(
    n_steps: int = N_STEPS,
    loss_func: AbstractLoss = simple_reach_loss(),
    validation_params: dict[str, Any] = TASK_EVAL_PARAMS['full'],
) -> SimpleReaches:
    return SimpleReaches(
        loss_func=loss_func,
        workspace=WORKSPACE, 
        n_steps=n_steps,
        **validation_params, 
    )
    
    
def get_train_pairs_by_disturbance_std(
    setup_task_model_pair: Callable, 
    model_hps: dict, 
    disturbance: dict, 
    key: PRNGKeyArray, 
    model_hps_update: Optional[dict] = None,
) -> TrainStdDict:
    if model_hps_update is None:
        model_hps_update = dict()
    
    disturbance_stds = disturbance['stds'][disturbance['type']]
    
    task_model_pairs = TrainStdDict(map_kwargs_to_dict(
        partial(
            setup_task_model_pair, 
            **model_hps | model_hps_update, 
            key=key,
        ),
        'disturbance_std',
        disturbance_stds,  
    ))
    return task_model_pairs


def setup_train_histories(
    models_tree,
    *,
    n_batches,
    batch_size,
    n_replicates,
    where_train_strs,
    save_model_parameters,
    readout_norm_value=None,
    readout_norm_loss_weight=None,
    key,
) -> dict[float, TaskTrainerHistory]:
    """Returns a skeleton PyTree for the training histories (losses, parameter history, etc.)
    
    Note that `init_task_trainer_history` depends on `task` to infer:
    
    1) The number and name of loss function terms;
    2) The structure of trial specs, in case `save_trial_specs is not None`.
    
    Here, neither of these are a concern since 1) we are always using the same 
    loss function for each set of saved/loaded models in this project, 2) `save_trial_specs is None`.
    """   
    # Assume that where funcs may be lists (normally defined as tuples, but retrieved through sqlite JSON)
    where_train = jt.map(
        attr_str_tree_to_where_func, 
        where_train_strs,
        is_leaf=is_type(list),
    )
    
    loss_func = simple_reach_loss()
    if readout_norm_loss_weight is not None:
        assert readout_norm_value is not None, (
            "readout_norm_value must be provided if readout_norm_loss_weight is not None"
        )
        loss_func_validation = loss_func + readout_norm_loss_weight * get_readout_norm_loss(readout_norm_value)
    else:
        loss_func_validation = loss_func
    
    return jt.map(
        lambda models: init_task_trainer_history(
            loss_func,
            n_batches,
            n_replicates,
            ensembled=True,
            ensemble_random_trials=False,
            save_model_parameters=jnp.array(save_model_parameters),
            save_trial_specs=None,
            batch_size=batch_size,
            loss_func_validation=loss_func_validation,
            model=models,
            where_train=where_train,  
        ),
        models_tree,
        is_leaf=is_module,
    )


def train_histories_hps_select(hps: dict) -> dict: 
    return dictmerge(
        subdict(hps['train'], [
            "n_batches",
            "batch_size",
            "where_train_strs",
            "save_model_parameters",
        ]),
        subdict(hps['model'], [
            "n_replicates",
            "disturbance_type",
            "feedback_delay_steps",
            "feedback_noise_std",
        ]),
    )


def get_latest_matching_file(directory: str, pattern: str) -> Optional[str]:
    """
    Returns the filename of the latest file in the given directory that matches the given pattern.

    The 'latest' file is determined by sorting the filenames in descending order.

    Arguments:
        directory: The directory path to search in.
        pattern: The pattern to match filenames against (e.g., 'A-*.json').

    Returns:
        The filename of the latest matching file, or None if no match is found.

    Raises:
        OSError: If there's an error reading the directory.
    """
    try:
        all_files = os.listdir(directory)
    except OSError as e:
        print(f"Error reading directory {directory}: {e}")
        return None

    matching_files = fnmatch.filter(all_files, pattern)

    if not matching_files:
        return None

    sorted_files = sorted(matching_files, reverse=True)

    return sorted_files[0]


def display_model_filechooser(path, filter_pattern='*.eqx',):
    """Display a file chooser interface for the files at `path` whose names satisfy `filter_pattern`.
    
    The default filename is the one that sorts last.
    """
    fc = FileChooser(path)
    fc.filter_pattern = filter_pattern
    fc.title = "Select model file:"
    params_widget = HTML("")
    
    default_filename = get_latest_matching_file(path, fc.filter_pattern)
    if default_filename is not None:
        fc.default_filename = default_filename

    def display_params(path, html_widget):
        with open(path, 'r') as f:
            params = json.load(f)
        params_str = eqx.tree_pformat(params, truncate_leaf=lambda x: isinstance(x, list) and len(x) > 10)
        html_widget.value = '<pre>' + params_str.replace(':\n', ':') + '</pre>'       
    
    def display_params_callback(fc: Optional[FileChooser]):
        if fc is None:
            return
        if fc.selected is None:
            raise RuntimeError("")
        return display_params(
            fc.selected.replace('trained_models.eqx', 'hyperparameters.json'),
            params_widget,
        )
        
    fc.register_callback(display_params_callback)

    display(fc, params_widget)
    
    return fc


def wait_for_value(variable, timeout: float = 3600):
    end_time = time.monotonic() + timeout
    while variable is None:
        if time.monotonic() > end_time:
            return False  # Timeout occurred
        time.sleep(0.1)
    return True


def choose_model_file(filter_pattern="*.eqx", timeout: float = 3600) -> str:
    """Displays a file chooser in the model directory until """
    fc = display_model_filechooser(MODELS_DIR, filter_pattern=filter_pattern)
    
    if wait_for_value(fc, timeout=timeout):
        assert fc.selected is not None
        return fc.selected
    else:
        return f"{fc.default_path}/{fc.default_filename}"


def find_unique_filepath(path: str | Path, search_string: str) -> Optional[Path]:
    """
    Returns the path of the unique file in a directory whose filename contains a given string.

    Arguments:
        directory: The path to the directory to search in.
        search_string: The string to search for in filenames.

    Returns:
        The path of the unique file if found, None otherwise.
    """
    # Convert directory to Path object if it's a string
    dir_path = Path(path) if isinstance(path, str) else path
    
    matching_files = [
        filename for filename in dir_path.iterdir()
        if filename.is_file() and search_string.lower() in filename.name.lower()
    ]

    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) == 0:
        print(f"No files found containing '{search_string}'.")
        return None
    else:
        print(f"Multiple files found containing '{search_string}':")
        for file in matching_files:
            print(file.name)
        return None


def filename_join(strs, joinwith="__"):
    """Helper for formatting filenames from lists of strings."""
    return joinwith.join(s for s in strs if s)


def set_model_noise(
    model, 
    noise_stds: dict[Literal['feedback', 'motor'], Optional[float]], 
    enable_noise: bool = True,
):
    """Change the system noise strength of a model."""
    get_noise_funcs = dict(
        feedback=lambda std: Normal(std=std),
        motor=lambda std: Multiplicative(Normal(std=std)) + Normal(std=1.8 * std),
    )
    
    noise_funcs = jt.map(
        lambda std, get_noise_func: get_noise_func(std),
        noise_stds, get_noise_funcs,
    )
    
    wheres = dict(
        feedback=lambda model: model.step.feedback_channels[0].noise_func,
        motor=lambda model: model.step.efferent_channel.noise_func,
    )
    
    pairs, LeafTuple = tree_zip_named(
        noise_func=noise_funcs,
        where=wheres, 
        is_leaf=is_module,
    )
    
    for noise_func, where in jt.leaves(pairs, is_leaf=is_type(LeafTuple)):
        model = eqx.tree_at(where, model, noise_func)
    
    if enable_noise:
        model = eqx.tree_at(
            lambda model: (
                model.step.feedback_channels[0].add_noise,
                model.step.efferent_channel.add_noise,
            ),
            model,
            (True, True),
        )
    
    return model
    

def setup_models_only(task_model_pair_setup_func, **kwargs):
    """Given a function that returns task-model pairs, just get the models."""
    task_model_pairs = task_model_pair_setup_func(**kwargs)
    _, models = tree_unzip(task_model_pairs)
    return models    


def setup_tasks_only(task_model_pair_setup_func, **kwargs):
    """Given a function that returns task-model pairs, just get the tasks."""
    task_model_pairs = task_model_pair_setup_func(**kwargs)
    tasks, _ = tree_unzip(task_model_pairs)
    return tasks


def convert_tasks_to_small(tasks):
    """Given a PyTree of tasks, return a matching PyTree where each task uses the small set of validation trials."""
    return jt.map(
        lambda task: eqx.tree_at(
            lambda task: tuple(getattr(task, k) for k in TASK_EVAL_PARAMS['small']),
            task, 
            tuple(TASK_EVAL_PARAMS['small'].values()),
        ),
        tasks,
        is_leaf=is_module,
    )


# When excluding models based on performance measures aside from loss, these are the ones we'll consider
MEASURES_TO_RATE = ('end_pos_error',)


def setup_replicate_info(models, n_replicates, *, key):
    """Returns a skeleton PyTree for loading the replicate info"""
    
    def models_tree_with_value(value):
        return jt.map(
            lambda _: value,
            models,
            is_leaf=is_module,
        )
        
    def get_measure_dict(value): 
        return dict.fromkeys(
            ("best_total_loss",) + MEASURES_TO_RATE,
            models_tree_with_value(value),
        )
    
    # For each piece of replicate info, we need a PyTree with the same structure as the model PyTree
    return {
        info_label: models_tree_with_value(value)
        for info_label, value in dict(
            best_save_idx=jnp.zeros(n_replicates, dtype=int),
            best_saved_iteration_by_replicate=[0] * n_replicates,
            losses_at_best_saved_iteration=jnp.zeros(n_replicates, dtype=float),
            losses_at_final_saved_iteration=jnp.zeros(n_replicates, dtype=float),
            readout_norm=jnp.zeros(n_replicates, dtype=float),
        ).items()
    } | dict(
        best_replicates=get_measure_dict(0),
        included_replicates=get_measure_dict(jnp.ones(n_replicates, dtype=bool)),
    )

    
def query_and_load_model(
    db_session,
    setup_task_model_pair: Callable,
    params_query: dict[str, Any],
    noise_stds: Optional[dict[Literal['feedback', 'motor'], Optional[float]]] = None,
    tree_inclusions: Optional[dict[type, Optional[Any | Sequence | Callable]]] = None,
    exclude_underperformers_by: Optional[str] = None,
    exclude_method: Literal['nan', 'remove', 'best-only'] = 'nan',
):
    """Query the models table in the project database and return the loaded and processed models.
    
    Arguments:
        db_session: The SQLAlchemy database session
        setup_task_model_pair: The function used to setup the task-model PyTree for this 
            part of the project.
        params_load: The parameters used to query the records of the model table of the database. 
            If more than one record matches, an error is raised.  
        tree_inclusions: Optionally, rules by which to include parts of dict nodes in the loaded 
            PyTree of models. Each rule's key is a dict node type in the PyTree,  
            and the respective values are the node key(s) which should be kept, or a callable that 
            returns true for the keys to be kept. If `None`, all nodes are kept as-is.
        exclude_underperformers_by: An optional key of a performance measure evaluated in 
            `post_training` by which to exclude model replicates. Excluded replicates will have 
            their parameters replaced with NaN in the arrays of the returned PyTree.
        exclude_method: Whether to index-out the included replicates ('remove'), replace their 
            model parameters with NaN ('nan'), or return only the single best replicate 
            ('best-only').
        
    Returns:
        model: The model PyTree
        model_info: The object mapping to the model's database record
        replicate_info: A dict of information about the model replicates
        n_replicates_included: The number of replicates not excluded (made NaN) from the model arrays
    """
    exclude_method_ = exclude_method.lower()
    
    model_info = get_model_record(
        db_session,
        has_replicate_info=True,
        **params_query,
    )
    
    if model_info is None:
        raise ValueError('No model with given parameters found in database!')
    
    assert model_info.replicate_info_path is not None, (
        "Model record's replicate_info_path is None, but has_replicate_info==True"
    )
    
    model: eqx.Module = load(
        model_info.path, partial(setup_models_only, setup_task_model_pair),
    )

    replicate_info: PyTree = load(
        model_info.replicate_info_path, partial(setup_replicate_info, model),
    )
    
    n_replicates_included = model_info.n_replicates
    
    if noise_stds is not None:
        # NOTE: Map over `model` as if it is type `PyTree[eqx.Module]`. This may be
        #       vestigial but it's also more general, and isn't costly, so I am leaving it as-is.
        model = jt.map(
            partial(
                set_model_noise, 
                noise_stds=noise_stds,
                enable_noise=True,
            ),
            model,
            is_leaf=is_module,
        )
    
    if tree_inclusions is not None:
        for dict_type, inclusion in tree_inclusions.items():
            if inclusion is not None:
                
                replace_func = lambda d, inclusion: subdict(d, inclusion)
                
                if isinstance(inclusion, Callable):
                    # Callables always result in sequence-like inclusions
                    inclusion = [
                        x for x in model_info.inclusion if all(inclusion(x))
                    ]
                elif isinstance(inclusion, str) or not isinstance(inclusion, Sequence):
                    # If not a Callable and not a Sequence, then assume we've 
                    # been given a single key to include
                    replace_func = lambda d, inclusion: d[inclusion]
                
                model, replicate_info = jt.map(
                    lambda d: replace_func(d, inclusion), 
                    (model, replicate_info),
                    is_leaf=is_type(dict_type),
                )
        
    if exclude_underperformers_by is not None:
        included_replicates = replicate_info['included_replicates'][exclude_underperformers_by]
        best_replicate = replicate_info['best_replicates'][exclude_underperformers_by]
        
        if exclude_method_ == 'nan':
            def include_func_nan(model, included, best): 
                return tree_set_scalar(model, jnp.nan, jnp.where(~included)[0])
            include_func = include_func_nan
        elif exclude_method_ == 'remove':
            def include_func_remove(model, included, best): 
                return take_model(model, jnp.where(included)[0])
            include_func = include_func_remove
        elif exclude_method_ == 'best-only':
            def include_func_best(model, included, best): 
                return take_model(model, best)
            include_func = include_func_best
        else:
            raise ValueError(f"Invalid exclude_method '{exclude_method_}'")
        
        model = jt.map(
            include_func,
            model, 
            included_replicates,
            best_replicate,
            is_leaf=is_module,
        )
        
        # print("\nReplicates included in analysis for each training condition:")
        # eqx.tree_pprint(jt.map(lambda x: jnp.where(x)[0], included_replicates), short_arrays=False)
    
        n_replicates_included = jt.map(lambda x: jnp.sum(x).item(), included_replicates)
        
        if any(n < 1 for n in jt.leaves(n_replicates_included)):
            raise ValueError("No replicates met inclusion criteria for at least one model variant")
    
    return model, model_info, replicate_info, n_replicates_included


def process_hps(hps: dict):
    """Resolve any dependencies and do any clean-up or validation of hyperparameters."""
    # Make a copy, to avoid in-place modification of the argument
    hps = dict(hps)

    # Update with missing arguments to `setup_task_model_pair` and `train_setup`, respectively
    hps['model'] |= dict(
        disturbance_type=hps['disturbance']['type'],
        intervention_scaleup_batches=(
            hps['train']['n_batches_baseline'],
            hps['train']['n_batches_baseline'] + hps['train']['n_scaleup_batches'],
        ),
    )
    hps['train']['n_batches'] = hps['train']['n_batches_baseline'] + hps['train']['n_batches_condition']
    hps['train']['save_model_parameters'] = get_iterations_to_save_model_parameters(
        hps['train']['n_batches']
    )
    
    return hps


TYPE_HP_KEY_MAPPING = {
    TrainingMethodDict: ("train", "train_method"),
    TrainStdDict: ("model", "disturbance_std"),
}


def update_hps_given_tree_path(hps: dict, path: tuple, types: Sequence) -> dict:
    hps = dict(hps)
    for node_key, type_ in zip(path, types):
        hps_key, hps_subkey = TYPE_HP_KEY_MAPPING[type_]
        hps[hps_key] = hps[hps_key] | {hps_subkey: node_key.key} 
    return hps
        

# TODO: Probably move this to `train.py`; when else do we save models like this?
def save_all_models(
    db_session,
    origin: str,
    models: PyTree[eqx.Module, 'T'], 
    hps: PyTree[dict, 'T'], 
    train_histories: Optional[PyTree[TaskTrainerHistory, 'T']] = None,
    **kwargs,
):
    """Saves a PyTree of models to disk, and the models table of the database.
    
    Optionally also stores all the training histories of the models.
    """
    model_records_flat = []
    
    path_vals, treedef = jtu.tree_flatten_with_path(models, is_leaf=is_module)
    
    for path, model in path_vals:
        idxs = [p.key for p in path]
        
        hps_i = index_multi(hps, *idxs)
        
        model_record = save_model_and_add_record(
            db_session,
            origin=origin,
            model=model,
            model_hyperparameters=hps_i['model'],
            other_hyperparameters=hps_i['train'],
            # Assume all the pytree levels are dicts
            train_history=index_multi(train_histories, *idxs),
            train_history_hyperparameters=train_histories_hps_select(hps_i),
            **kwargs,
        )
        
        model_records_flat.append(model_record)
    
    model_records = jt.unflatten(treedef, model_records_flat)
    return model_records


