import argparse
from functools import partial
import logging
import os
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
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

from rnns_learn_robust_motor_policies.misc import load_from_json, write_to_json
from rnns_learn_robust_motor_policies.part1_setup import setup_models, setup_train_histories


logging.basicConfig(
    # format='%(asctime)s - %(levelname)-8s - %(message)s', 
    level=logging.INFO, 
    handlers=[RichHandler(level="NOTSET")],
)
logger = logging.getLogger('rich')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models for part 1.")
    parser.add_argument("--directory", default="../models/", help="Directory to search for files")
    parser.add_argument("--filename_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--exclude_percentile", default=75, type=float, help="Percentile loss for replicate exclusion")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    model_paths = [
        str(directory / filename) for filename in os.listdir(directory)
        if args.filename_pattern in filename
    ]
    
    logger.info(f"Found {len(model_paths)} model files")
    
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(model_paths))
        
        for path in model_paths:
            
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
            where_train = attr_str_tree_to_where_func(hyperparameters['where_train_strs'])
            n_replicates = hyperparameters['n_replicates']        
            save_model_parameters = jnp.array(hyperparameters['save_model_parameters'])
            
            # Get info about the best replicates, and which replicates to exclude 
            best_save_idx_by_replicate = jt.map(
                lambda history: jnp.argmin(
                    history.loss.total[save_model_parameters], 
                    axis=0,
                ), 
                train_histories, 
                is_leaf=is_module,
            )

            best_saved_iteration_by_replicate = jt.map(
                lambda idx: save_model_parameters[idx].tolist(), 
                best_save_idx_by_replicate, 
            )
            
            loss_at_best_saved_iteration_by_replicate = jt.map(
                lambda history, saved_iterations: (
                    history.loss.total[jnp.array(saved_iterations), jnp.arange(n_replicates)]
                ),
                train_histories, best_saved_iteration_by_replicate,
                is_leaf=is_module,
            )

            loss_at_final_iteration_by_replicate = jt.map(
                lambda history: history.loss.total[-1],
                train_histories,
                is_leaf=is_module,
            )
            
            best_replicate_by_loss = jt.map(
                lambda best_losses: jnp.argmin(best_losses).item(), 
                loss_at_best_saved_iteration_by_replicate,
                is_leaf=is_module,
            )
            
            exclusion_loss_bound = jt.map(
                lambda losses: jnp.percentile(losses, args.exclude_percentile).item(),
                loss_at_best_saved_iteration_by_replicate,
            )

            included_replicates = jt.map(
                lambda losses, bound: losses < bound,
                loss_at_best_saved_iteration_by_replicate, 
                exclusion_loss_bound,
            )
            
            write_to_json(
                dict(
                    best_saved_iteration_by_replicate=best_saved_iteration_by_replicate,
                    loss_at_best_saved_iteration_by_replicate=loss_at_best_saved_iteration_by_replicate,
                    best_replicate_by_loss=best_replicate_by_loss,
                    included_replicates=included_replicates,    
                ),
                path.replace('trained_models.eqx', 'extras.json'),
            )
            
            # Save the best model parameters (for each replicate) to a model file
            best_saved_parameters_by_replicate = tree_stack([
                jt.map(
                    lambda train_history, best_idx_by_replicate: tree_take_multi(
                        train_history.model_parameters, 
                        [int(best_idx_by_replicate[i]), i], 
                        [0, 1],
                    ),
                    train_histories, best_save_idx_by_replicate,
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
                best_saved_parameters_by_replicate,
                is_leaf=is_module,
            )
            
            save(
                path.replace(args.filename_pattern, args.filename_pattern.replace('.eqx', '_best_params.eqx')),
                models_with_best_parameters,
                hyperparameters=model_hyperparameters,
            )
            
            logger.info(f"Processed {path}.")
            progress.update(task, advance=1)