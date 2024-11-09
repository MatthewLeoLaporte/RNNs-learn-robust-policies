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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.progress import Progress
from rich.logging import RichHandler

from feedbax import (
    is_module, 
    is_type,
    load, 
    load_with_hyperparameters, 
    save, 
    tree_stack, 
    tree_take_multi,
)
from feedbax.misc import attr_str_tree_to_where_func
import feedbax.plot as fbplt
import feedbax.plotly as fbp 
from feedbax.train import TaskTrainerHistory
from feedbax._tree import tree_labels

from rnns_learn_robust_motor_policies.misc import load_from_json, write_to_json
from rnns_learn_robust_motor_policies.plot_utils import get_savefig_func
from rnns_learn_robust_motor_policies.setup_utils import filename_join as join
from rnns_learn_robust_motor_policies.part1_setup import setup_models as setup_models_p1
from rnns_learn_robust_motor_policies.part2_setup import TrainStdDict
from rnns_learn_robust_motor_policies.part2_setup import setup_models as setup_models_p2
from rnns_learn_robust_motor_policies.setup_utils import setup_train_histories


logging.basicConfig(
    format='(%(module)-20s) %(message)s', 
    level=logging.INFO, 
    handlers=[RichHandler(level="NOTSET")],
)
logger = logging.getLogger('rich')


SETUP_FUNCS = {
    '1-1': setup_models_p1,
    '2-1': setup_models_p2,
}


def find_model_paths(directory: str, filename_pattern: str) -> list[str]:
    """Returns paths to all model files matching the pattern in the directory."""
    return [
        str(Path(directory) / filename) 
        for filename in os.listdir(directory)
        if filename_pattern in filename
    ]


def load_data(path: str, setup_func: Callable):
    """Loads models, hyperparameters and training histories from files."""
    models, model_hyperparameters = load_with_hyperparameters(
        path, setup_func,
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


def get_loss_history_figures(train_histories, disturbance_type, n_replicates):
    # TODO: these plots are large/sluggish for `n_steps` >> 1e3. Can probably downsample; start with every 1 to 1000, then every 10 to 10000.
    
    def get_fig(history, disturbance_std):
        label = f"{n_replicates}-reps"
        
        p1, _ = fbplt.loss_history(history.loss)
        # p2, _ = fbplt.loss_mean_history(history.loss)
        p3 = fbp.loss_history(history.loss)
        
        return {
            join([label, 'replicates']): p1, 
            label: p3,
        }
    
    def get_variant_figs(variant_histories, variant_label):
        return {
            f"{disturbance_type}__std-{disturbance_std}": get_fig(history, disturbance_std)
            for disturbance_std, history in variant_histories.items()
        }
    
    return jt.map(
        get_variant_figs, 
        train_histories, 
        tree_labels(train_histories, is_leaf=is_type(TrainStdDict)),
        is_leaf=is_type(TrainStdDict),
    )


def get_replicate_loss_distribution_figures(
    losses_at_best_saved_iteration, 
    losses_at_final_saved_iteration,
    n_replicates,
):
    
    def get_fig(losses_by_replicate, label):
        df = pd.DataFrame(losses_by_replicate).reset_index().melt(id_vars='index')
        df["index"] = df["index"].astype(str)

        fig = go.Figure()

        strips = px.scatter(
            df,
            x='variable',
            y='value',
            color="index",
            labels=dict(x="Train disturbance std.", y=f"{label} batch total loss"),
            title=f"{label} total loss across training by disturbance std.",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            # stripmode='overlay',
        )
        
        strips.update_traces(
            marker_size=10,
            marker_symbol='circle-open',
            marker_line_width=3,
        )
        
        # strips.for_each_trace(
        #     lambda trace: trace.update(x=list(trace.x))
        # )

        violins = [
            go.Violin(
                x=[disturbance_std] * n_replicates,
                y=losses,
                # box_visible=True,
                line_color='black',
                meanline_visible=True,
                fillcolor='lightgrey',
                opacity=0.6,
                name=f"{disturbance_std}",
                showlegend=False,   
                spanmode='hard',  
            )
            for disturbance_std, losses in losses_by_replicate.items()
        ]
        
        fig.add_traces(violins)
        fig.add_traces(strips.data)

        fig.update_layout(
            xaxis_type='category',
            width=800,
            height=500,
            xaxis_title="Train disturbance std.",
            yaxis_title=f"{label} total loss",
            # xaxis_range=[-0.5, len(disturbance_stds) + 0.5],
            # xaxis_tickvals=np.linspace(0,1.2,4),
            # yaxis_type='log',
            violingap=0.1,
            # showlegend=False,
            legend_title='Replicate',
            legend_tracegroupgap=4,
            # violinmode='overlay',  
            barmode='overlay',
            # boxmode='group',
        )
        
        return fig
       
    def get_variant_figs(
        variant_losses_at_best_saved_iteration, 
        variant_losses_at_final_saved_iteration, 
        variant_label,
    ):
        return {
            f"{label.lower()}": get_fig(losses_by_replicate, label) 
            for label, losses_by_replicate in {
                "Best": variant_losses_at_best_saved_iteration,
                "Final": variant_losses_at_final_saved_iteration,
            }.items() 
        }

    return jt.map(
        get_variant_figs,
        losses_at_best_saved_iteration,
        losses_at_final_saved_iteration,
        tree_labels(losses_at_best_saved_iteration, is_leaf=is_type(TrainStdDict)),
        is_leaf=is_type(TrainStdDict),
    )


def save_training_figures(
    figs_dir,
    train_histories, 
    disturbance_type, 
    n_replicates,
    losses_at_best_saved_iteration,
    losses_at_final_saved_iteration,
):
    all_figs = {
        'loss_history': get_loss_history_figures(train_histories, disturbance_type, n_replicates),
        'loss_dist_over_replicates': get_replicate_loss_distribution_figures(
            losses_at_best_saved_iteration, 
            losses_at_final_saved_iteration,
            n_replicates,
        ),
    }
    
    savefig = get_savefig_func(figs_dir)
    
    for fig_subdir, figs in all_figs.items():
        jt.map(
            lambda fig, label: savefig(fig, label, subdir=fig_subdir),
            figs, 
            tree_labels(figs, is_leaf=is_type(go.Figure)),
            is_leaf=is_type(go.Figure),
        )
        logger.info(f"Saved figure set to {figs_dir}/{fig_subdir}")
    

def process_model_file(path: str, n_std_exclude: float, filename_pattern: str, figs_base_dir: str, nb_id) -> None:
    """Processes a single model file, computing and saving replicate info and best parameters."""
    models, model_hyperparameters, train_histories, hyperparameters = load_data(path, SETUP_FUNCS[nb_id])
    
    where_train = attr_str_tree_to_where_func(hyperparameters['where_train_strs'])
    disturbance_type = hyperparameters['disturbance_type']
    suffix = hyperparameters['suffix']
    n_replicates = hyperparameters['n_replicates']        
    save_model_parameters = jnp.array(hyperparameters['save_model_parameters'])
    
    best_save_idx, best_saved_iterations, losses_at_best_saved_iteration = \
        get_best_iterations_and_losses(
            train_histories, save_model_parameters, n_replicates
        )
    
    best_replicate, included_replicates = compute_replicate_info(
        losses_at_best_saved_iteration, n_std_exclude
    )
    
    losses_at_final_saved_iteration = jt.map(
        lambda history: history.loss.total[-1],
        train_histories,
        is_leaf=is_module,
    )
    
    write_to_json(
        dict(
            best_saved_iteration_by_replicate=best_saved_iterations,
            losses_at_best_saved_iteration=losses_at_best_saved_iteration,
            losses_at_final_saved_iteration=losses_at_final_saved_iteration,
            best_replicate_by_loss=best_replicate,
            included_replicates=included_replicates,    
        ),
        path.replace('trained_models.eqx', 'extras.json'),
    )
    
    save_best_models(
        models, 
        train_histories, 
        best_save_idx, 
        n_replicates, 
        where_train, 
        path, 
        filename_pattern, 
        model_hyperparameters,
    )
    
    figs_dir = Path(f'{figs_base_dir}/{nb_id}/{suffix}')
    
    save_training_figures(
        figs_dir,
        train_histories, 
        disturbance_type, 
        n_replicates, 
        losses_at_best_saved_iteration, 
        losses_at_final_saved_iteration,
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models for part 1.")
    parser.add_argument("--model_dir", default="./models/", help="Directory to search for files")
    parser.add_argument("--figs_base_dir", default="./figures/", help="Base directory for saving figures")
    parser.add_argument("--filename_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--n_std_exclude", default=1, type=float, help="Percentile loss for replicate exclusion")
    args = parser.parse_args()
    
    model_paths = find_model_paths(args.model_dir, args.filename_pattern)
    logger.info(f"Found {len(model_paths)} model files")
    
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(model_paths))
        for path in model_paths:
            nb_id = path.split('/')[-1].split('_')[0]
            process_model_file(path, args.n_std_exclude, args.filename_pattern, args.figs_base_dir, nb_id)
            logger.info(f"Processed {path}.")
            progress.update(task, advance=1)