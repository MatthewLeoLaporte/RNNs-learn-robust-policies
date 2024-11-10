import argparse
from collections.abc import Callable
from functools import partial
import logging
import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
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
    tree_unzip,
)
from feedbax.misc import attr_str_tree_to_where_func
import feedbax.plot as fbplt
import feedbax.plotly as fbp 
from feedbax.train import TaskTrainerHistory
from feedbax._tree import tree_labels

from rnns_learn_robust_motor_policies.misc import load_from_json, write_to_json
from rnns_learn_robust_motor_policies.plot_utils import get_savefig_func
from rnns_learn_robust_motor_policies.part1_setup import (
    setup_task_model_pairs as setup_task_model_pairs_p1
)
from rnns_learn_robust_motor_policies.part2_setup import (
    setup_task_model_pairs as setup_task_model_pairs_p2
)
from rnns_learn_robust_motor_policies.part2_setup import TrainStdDict
from rnns_learn_robust_motor_policies.setup_utils import (
    setup_train_histories,
    setup_models_only,
    setup_tasks_only,
    filename_join as join,
)
from rnns_learn_robust_motor_policies.state_utils import (
    get_aligned_vars,
    get_pos_endpoints,
    vmap_eval_ensemble,
)


logging.basicConfig(
    format='(%(name)-20s) %(message)s', 
    level=logging.INFO, 
    handlers=[RichHandler(level="NOTSET")],
)
logger = logging.getLogger('rich')


SETUP_FUNCS = {
    '1-1': setup_task_model_pairs_p1,
    '2-1': setup_task_model_pairs_p2,
}

# Number of trials to evaluate when deciding which replicates to exclude
N_TRIALS_VAL = 5


def find_model_paths(directory: str, filename_pattern: str) -> list[str]:
    """Returns paths to all model files matching the pattern in the directory."""
    return [
        str(Path(directory) / filename) 
        for filename in os.listdir(directory)
        if filename_pattern in filename
    ]


def load_data(path: str, nb_id: str):
    """Loads models, hyperparameters and training histories from files."""
    models, model_hyperparameters = load_with_hyperparameters(
        path, partial(setup_models_only, SETUP_FUNCS[nb_id]),
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


def get_best_and_included(measure, n_std_exclude=2):
    best_idx = jnp.argmin(measure).item()
    bound = (measure[best_idx] + n_std_exclude * measure.std()).item()
    included = measure < bound
    return best_idx, included


def end_position_error(pos, eval_reach_length=1, last_n_steps=10):
    # Since the data is aligned, the goal is always at the same position
    goal_pos = jnp.array([eval_reach_length, 0])
    error = jnp.mean(jnp.linalg.norm(pos[..., -last_n_steps:, :] - goal_pos, axis=-1), axis=-1)
    return error


def get_measures_to_rate(models, tasks):
    all_states = jt.map(
        lambda model, task: vmap_eval_ensemble(model, task, N_TRIALS_VAL, jr.PRNGKey(0)),
        models, tasks,
        is_leaf=is_module,
    )
    # Assume all the tasks have the same reach endpoints and reach length
    example_task = jt.leaves(tasks, is_leaf=is_module)[0]
    pos_endpoints = get_pos_endpoints(example_task.validation_trials)
    aligned_pos = get_aligned_vars(
        all_states,
        lambda states, endpoints: states.mechanics.effector.pos - pos_endpoints[0][..., None, :],
        pos_endpoints,
    )
    end_pos_errors = jt.map(
        partial(end_position_error, eval_reach_length=example_task.eval_reach_length), 
        aligned_pos,
    )
    mean_end_pos_errors = jt.map(
        lambda x: jnp.mean(x, axis=(0, -1)),  # eval & condition, but not replicate
        end_pos_errors,
    )
    
    return dict(
        end_pos_error=mean_end_pos_errors,
    )
    

MEASURES_TO_RATE = ('end_pos_error',)


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
        path.replace('.eqx', '_best_params.eqx'),
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


def compute_replicate_info(
    models,
    tasks,
    train_histories, 
    save_model_parameters, 
    n_replicates, 
    n_std_exclude,
):
    best_save_idx, best_saved_iterations, losses_at_best_saved_iteration = \
        get_best_iterations_and_losses(
            train_histories, save_model_parameters, n_replicates
        )
    
    # Rate the best total loss, but also some other measures
    measures = dict(
        best_total_loss=losses_at_best_saved_iteration,
        **get_measures_to_rate(models, tasks),
    )
    
    best_replicates, included_replicates = tree_unzip(jt.map(
        partial(get_best_and_included, n_std_exclude=n_std_exclude),
        measures,
    ))
    
    losses_at_final_saved_iteration = jt.map(
        lambda history: history.loss.total[-1],
        train_histories,
        is_leaf=is_module,
    )
    
    return dict(
        best_save_idx=best_save_idx,
        best_saved_iteration_by_replicate=best_saved_iterations,
        losses_at_best_saved_iteration=losses_at_best_saved_iteration,
        losses_at_final_saved_iteration=losses_at_final_saved_iteration,
        best_replicates=best_replicates,
        included_replicates=included_replicates,
    )   
    
    
def setup_replicate_info(models, disturbance_stds, n_replicates, *, key):
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
        ).items()
    } | dict(
        best_replicates=get_measure_dict(0),
        included_replicates=get_measure_dict(jnp.ones(n_replicates, dtype=bool)),
    )
    

    
    aaa 
    
    eqx.tree_pprint(aaa)
    
    return aaa


def process_model_file(path: str, n_std_exclude: float, filename_pattern: str, figs_base_dir: str, nb_id) -> None:
    """Processes a single model file, computing and saving replicate info and best parameters."""
    models, model_hyperparameters, train_histories, hyperparameters = load_data(path, nb_id)
    
    where_train = attr_str_tree_to_where_func(hyperparameters['where_train_strs'])
    disturbance_type = hyperparameters['disturbance_type']
    suffix = hyperparameters['suffix']
    n_replicates = hyperparameters['n_replicates']        
    save_model_parameters = jnp.array(hyperparameters['save_model_parameters'])
    
    # Evaluate each model on its respective validation task
    tasks = setup_tasks_only(SETUP_FUNCS[nb_id], key=jr.PRNGKey(0), **model_hyperparameters)
    
    replicate_info = compute_replicate_info(
        models,
        tasks,
        train_histories, 
        save_model_parameters, 
        n_replicates, 
        n_std_exclude, 
    )
    
    save(
        path.replace('trained_models', 'replicate_info'),
        replicate_info,
        hyperparameters=dict(
            disturbance_stds=hyperparameters['disturbance_stds'],
            n_replicates=n_replicates,
        ),
    )
    
    save_best_models(
        models, 
        train_histories, 
        replicate_info['best_save_idx'], 
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
        replicate_info['losses_at_best_saved_iteration'], 
        replicate_info['losses_at_final_saved_iteration'],
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models for part 1.")
    parser.add_argument("--model_dir", default="./models/", help="Directory to search for files")
    parser.add_argument("--figs_base_dir", default="./figures/", help="Base directory for saving figures")
    parser.add_argument("--filename_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--n_std_exclude", default=2, type=float, help="Mark replicates this many stds above the best as to-be-excluded")
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