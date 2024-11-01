import argparse
from collections.abc import Callable
from functools import partial
import logging
import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Int, Bool, Float, PyTree
import numpy as np
from rich.progress import Progress
from rich.logging import RichHandler

from feedbax import (
    is_module, 
    load, 
    load_with_hyperparameters, 
    save, 
    tree_stack, 
    tree_take_multi,
)
from feedbax.misc import attr_str_tree_to_where_func
from feedbax.train import TaskTrainerHistory

from rnns_learn_robust_motor_policies.misc import load_from_json, write_to_json
from rnns_learn_robust_motor_policies.part1_setup import setup_models, setup_train_histories


logging.basicConfig(
    format='(%(module)-20s) %(message)s', 
    level=logging.INFO, 
    handlers=[RichHandler(level="NOTSET")],
)
logger = logging.getLogger('rich')


def find_model_paths(directory: str, filename_pattern: str) -> list[str]:
    """Returns paths to all model files matching the pattern in the directory."""
    return [
        str(Path(directory) / filename) 
        for filename in os.listdir(directory)
        if filename_pattern in filename
    ]


def load_data(path: str):
    """Loads models, hyperparameters and training histories from files."""
    models, model_hyperparameters = load_with_hyperparameters(
        path, setup_models,
    )
    train_histories = load(
        path.replace('trained_models', 'train_histories'),
        partial(setup_train_histories, models),
    )
    hyperparameters = load_from_json(
        path.replace('trained_models.eqx', 'hyperparameters.json')
    )
    return models, model_hyperparameters, train_histories, hyperparameters


def get_best_iterations_and_losses(
    train_histories: PyTree[TaskTrainerHistory], 
    save_model_parameters: Array, 
    n_replicates: int
):
    """Computes best iterations and corresponding losses for each replicate."""
    best_save_idx = jt.map(
        lambda history: jnp.argmin(
            history.loss.total[save_model_parameters], 
            axis=0,
        ), 
        train_histories, 
        is_leaf=is_module,
    )
    
    best_saved_iterations = jt.map(
        lambda idx: save_model_parameters[idx].tolist(), 
        best_save_idx, 
    )
    
    losses_at_best_saved_iteration = jt.map(
        lambda history, saved_iterations: (
            history.loss.total[jnp.array(saved_iterations), jnp.arange(n_replicates)]
        ),
        train_histories, best_saved_iterations,
        is_leaf=is_module,
    )
    
    return best_save_idx, best_saved_iterations, losses_at_best_saved_iteration


def compute_replicate_info(
    losses_at_best_saved_iteration: PyTree[Float[Array, "replicate"]], 
    n_std_exclude: float = 2.0,
) -> tuple[PyTree[int], PyTree[Array]]:
    """Identifies best replicates and which replicates to include based on loss distribution."""
    best_replicate = jt.map(
        lambda best_losses: jnp.argmin(best_losses).item(), 
        losses_at_best_saved_iteration,
        is_leaf=is_module,
    )
    
    exclusion_bound = jt.map(
        lambda losses: (losses.mean() + n_std_exclude * losses.std()).item(),
        losses_at_best_saved_iteration,
    )
    
    included_replicates = jt.map(
        lambda losses, bound: losses < bound,
        losses_at_best_saved_iteration, 
        exclusion_bound,
    )
    
    return best_replicate, included_replicates


def save_best_models(
    models: PyTree[eqx.Module],
    train_histories: PyTree[TaskTrainerHistory],
    best_save_idx: PyTree[Int[Array, "replicate"]],
    n_replicates: int,
    where_train: Callable[[eqx.Module], PyTree[Array]],
    path: str,
    filename_pattern: str,
    model_hyperparameters: dict[str, Any],
) -> None:
    """Serializes models with the best parameters for each replicate and training condition."""
    best_saved_parameters = tree_stack([
        jt.map(
            lambda train_history, best_idx_by_replicate: tree_take_multi(
                train_history.model_parameters, 
                [int(best_idx_by_replicate[i]), i], 
                [0, 1],
            ),
            train_histories, best_save_idx,
            is_leaf=is_module,
        )
        for i in range(n_replicates)
    ])
    
    models_with_best_parameters = jt.map(
        lambda model, best_params: eqx.tree_at(
            where_train,
            model, 
            where_train(best_params),
        ),
        models, 
        best_saved_parameters,
        is_leaf=is_module,
    )
    
    save(
        path.replace(filename_pattern, filename_pattern.replace('.eqx', '_best_params.eqx')),
        models_with_best_parameters,
        hyperparameters=model_hyperparameters,
    )


def process_model_file(path: str, n_std_exclude: float, filename_pattern: str) -> None:
    """Processes a single model file, computing and saving replicate info and best parameters."""
    models, model_hyperparameters, train_histories, hyperparameters = load_data(path)
    
    where_train = attr_str_tree_to_where_func(hyperparameters['where_train_strs'])
    n_replicates = hyperparameters['n_replicates']        
    save_model_parameters = jnp.array(hyperparameters['save_model_parameters'])
    
    best_save_idx, best_saved_iterations, losses_at_best_saved_iteration = \
        get_best_iterations_and_losses(
            train_histories, save_model_parameters, n_replicates
        )
    
    best_replicate, included_replicates = compute_replicate_info(
        losses_at_best_saved_iteration, n_std_exclude
    )
    
    write_to_json(
        dict(
            best_saved_iteration_by_replicate=best_saved_iterations,
            loss_at_best_saved_iteration_by_replicate=losses_at_best_saved_iteration,
            best_replicate_by_loss=best_replicate,
            included_replicates=included_replicates,    
        ),
        path.replace('trained_models.eqx', 'extras.json'),
    )
    
    save_best_models(
        models, train_histories, best_save_idx, n_replicates, 
        where_train, path, filename_pattern, model_hyperparameters
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models for part 1.")
    parser.add_argument("--directory", default="../models/", help="Directory to search for files")
    parser.add_argument("--filename_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--n_std_exclude", default=1, type=float, help="Percentile loss for replicate exclusion")
    args = parser.parse_args()
    
    model_paths = find_model_paths(args.directory, args.filename_pattern)
    logger.info(f"Found {len(model_paths)} model files")
    
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(model_paths))
        for path in model_paths:
            process_model_file(path, args.n_std_exclude, args.filename_pattern)
            logger.info(f"Processed {path}.")
            progress.update(task, advance=1)