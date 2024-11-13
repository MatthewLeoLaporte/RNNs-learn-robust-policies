import argparse
from collections.abc import Callable
from functools import partial
import logging
import os
from pathlib import Path
from sqlalchemy.orm import Session
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

from rnns_learn_robust_motor_policies import FIGS_BASE_DIR
from rnns_learn_robust_motor_policies.database import (
    ModelRecord, 
    MODEL_RECORD_BASE_ATTRS,
    get_db_session, 
    query_model_records,
    save_model_and_add_record,
)
from rnns_learn_robust_motor_policies.plot_utils import get_savefig_func
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
from rnns_learn_robust_motor_policies.train_setup_part1 import (
    setup_task_model_pairs as setup_task_model_pairs_p1
)
from rnns_learn_robust_motor_policies.train_setup_part2 import (
    setup_task_model_pairs as setup_task_model_pairs_p2
)
from rnns_learn_robust_motor_policies.types import TrainStdDict


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


def load_data(model_record: ModelRecord):
    """Loads models, hyperparameters and training histories from files."""
    # Load model and associated data
    notebook_id = str(model_record.notebook_id)
    model_path = str(model_record.model_path)
    train_history_path = str(model_record.train_history_path)
    
    if not os.path.exists(model_path) or not os.path.exists(train_history_path):
        logger.error(f"Model or training history file not found for {model_record.hash}")
        return None, None, None, None
    
    models, model_hyperparameters = load_with_hyperparameters(
        model_path, 
        partial(setup_models_only, SETUP_FUNCS[notebook_id]),
    )
    train_histories, train_history_hyperparameters = load_with_hyperparameters(
        train_history_path,
        partial(setup_train_histories, models),
    )
    
    return (
        models, 
        model_hyperparameters, 
        train_histories, 
        train_history_hyperparameters,
    )


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


def get_best_models(
    models: PyTree[eqx.Module, 'T'],
    train_histories: PyTree[TaskTrainerHistory],
    best_save_idx: PyTree[Int[Array, "replicate"]],
    n_replicates: int,
    where_train: Callable[[eqx.Module], PyTree[Array]],
) -> PyTree[eqx.Module, 'T']:
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
    
    return models_with_best_parameters


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
    
    readout_norm = jt.map(
        lambda model: jnp.linalg.norm(model.step.net.readout.weight, axis=(-2, -1), ord='fro'),
        models,
        is_leaf=is_module,        
    )
    
    return dict(
        best_save_idx=best_save_idx,
        best_saved_iteration_by_replicate=best_saved_iterations,
        losses_at_best_saved_iteration=losses_at_best_saved_iteration,
        losses_at_final_saved_iteration=losses_at_final_saved_iteration,
        best_replicates=best_replicates,
        included_replicates=included_replicates,
        readout_norm=readout_norm,
    )   
    
    
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

def process_model_record(
    session: Session,
    model_record: ModelRecord,
    n_std_exclude: float,
    figs_base_dir: Path,
) -> None:
    """Process a single model record, updating it with best parameters and replicate info."""
    
    notebook_id = str(model_record.notebook_id)
    where_train = attr_str_tree_to_where_func(model_record.where_train_strs)
    disturbance_type = str(model_record.disturbance_type)
    n_replicates = int(model_record.n_replicates)       
    save_model_parameters = jnp.array(model_record.save_model_parameters)
    
    models, model_hyperparams, train_histories, train_history_hyperparams = load_data(model_record)
    
    if None in (models, model_hyperparams, train_histories, train_history_hyperparams):
        return
    
    # Evaluate each model on its respective validation task
    tasks = setup_tasks_only(
        SETUP_FUNCS[notebook_id], 
        key=jr.PRNGKey(0), 
        **model_hyperparams,
    )
    
    # Compute replicate info``
    replicate_info = compute_replicate_info(
        models,
        tasks,
        train_histories, 
        save_model_parameters, 
        n_replicates, 
        n_std_exclude, 
    )
    
    # Create models with best parameters
    best_models = get_best_models(
        models, 
        train_histories, 
        replicate_info['best_save_idx'], 
        n_replicates, 
        where_train,
    )
    
    try:
        # Save new model file with best parameters and get new record
        record_hyperparameters = {
            key: getattr(model_record, key)
            for key in model_record.__table__.columns.keys()
            if key not in MODEL_RECORD_BASE_ATTRS
        }
        
        new_record = save_model_and_add_record(
            session,
            str(model_record.notebook_id),
            best_models,
            model_hyperparams,
            record_hyperparameters | dict(n_std_exclude=n_std_exclude),
            train_history=train_histories,
            train_history_hyperparameters=train_history_hyperparams,
            replicate_info=replicate_info,
            replicate_info_hyperparameters=dict(n_replicates=n_replicates),
        )
        
        # Delete old files if their paths changed
        paths = {
            'model': (model_record.model_path, new_record.model_path),
            'train_history': (model_record.train_history_path, new_record.train_history_path),
            # Replicate info should only change if we re-run `post_training` with different parameters
            'replicate_info': (model_record.replicate_info_path, new_record.replicate_info_path),
        }
        
        for key, (old_path, new_path) in paths.items():
            if old_path is not None and str(old_path) != str(new_path):
                Path(str(old_path)).unlink()
                logger.info(f"Deleted old {key} file: {old_path}")
        
        # Delete the old record if the hash changed
        # (If the hash remained the same, `save_model_and_add_record` has already dealt with it)
        if str(new_record.hash) != str(model_record.hash):    
            session.delete(model_record)
            session.commit()
        
    except Exception as e:
        # If anything fails, rollback and restore original record
        session.rollback()
        logger.error(f"Failed to process model {model_record.hash}: {e}")
        raise 
    
    # Save training figures
    figs_dir = figs_base_dir / f'{model_record.notebook_id}'
    save_training_figures(
        figs_dir,
        train_histories, 
        disturbance_type, 
        n_replicates, 
        replicate_info['losses_at_best_saved_iteration'], 
        replicate_info['losses_at_final_saved_iteration'],
    )
    
    logger.info(f"Processed model {model_record.hash}")
    
    
def main(n_std_exclude: float = 2.0):
    """Process all models in database."""
    session = get_db_session("models")
    
    # Get all model records
    model_records = query_model_records(session)
    logger.info(f"Found {len(model_records)} model records")
    
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(model_records))
        for model_record in model_records:
            try:
                process_model_record(
                    session,
                    model_record,
                    n_std_exclude,
                    FIGS_BASE_DIR,
                )
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Skipping model {model_record.hash} due to error: {e}")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models.")
    parser.add_argument("--n_std_exclude", default=2, type=float, help="Mark replicates this many stds above the best as to-be-excluded")
    args = parser.parse_args()
    
    main(args.n_std_exclude)