import argparse
from collections.abc import Callable
from functools import partial
import logging
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from sqlalchemy.orm import Session

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Int, PyTree, Shaped
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.progress import Progress
from rich.logging import RichHandler

from feedbax import (
    is_module, 
    is_type,
    load_with_hyperparameters, 
    tree_stack, 
    tree_take_multi,
    tree_unzip,
)
from feedbax.misc import attr_str_tree_to_where_func
import feedbax.plotly as fbp 
from feedbax.train import TaskTrainerHistory, WhereFunc, _get_trainable_params_superset
from feedbax._tree import filter_spec_leaves, tree_labels

from rnns_learn_robust_motor_policies.database import (
    Base,
    ModelRecord, 
    MODEL_RECORD_BASE_ATTRS,
    add_evaluation,
    add_evaluation_figure,
    get_db_session, 
    query_model_records,
    record_to_dict,
    save_model_and_add_record,
)
from rnns_learn_robust_motor_policies.setup_utils import (
    setup_train_histories,
    setup_models_only,
    setup_tasks_only,
)
from rnns_learn_robust_motor_policies.state_utils import (
    get_aligned_vars,
    get_pos_endpoints,
    vmap_eval_ensemble,
)
from rnns_learn_robust_motor_policies.train_setup_part1 import (
    setup_task_model_pairs as setup_task_model_pairs_p1,
    # setup_train_histories,
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


# The setup/deserialisation depends on where/how the model was trained
SETUP_FUNCS = {
    '1-1': setup_task_model_pairs_p1,
    '2-1': setup_task_model_pairs_p2,
}


# Number of trials to evaluate when deciding which replicates to exclude
N_TRIALS_VAL = 5


def load_data(model_record: ModelRecord):
    """Loads models, hyperparameters and training histories from files."""
    # Load model and associated data
    origin = str(model_record.origin)
    
    if not model_record.path.exists() or not model_record.train_history_path.exists():
        logger.error(f"Model or training history file not found for {model_record.hash}")
        return
    
    models, model_hyperparameters = load_with_hyperparameters(
        model_record.path, 
        partial(setup_models_only, SETUP_FUNCS[origin]),
    )
    logger.debug(f"Loaded model hyperparameters: {model_hyperparameters}")
    
    train_histories, train_history_hyperparameters = load_with_hyperparameters(
        model_record.train_history_path,
        partial(setup_train_histories, models),
    )
    logger.debug(f"Loaded train history hyperparameters: {train_history_hyperparameters}")
    
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


def _get_most_recent_idxs(idxs: Sequence[int], max_idx: int) -> Any:
    """Returns the value for the largest key less than or equal to `idx`."""
    keys = jnp.array(sorted(idxs))
    key_idxs = np.searchsorted(keys, np.arange(max_idx + 1), side='right')
    return keys[key_idxs - 1]


def get_best_models(
    model_record: ModelRecord,
    models: PyTree[eqx.Module, 'T'],
    train_histories: PyTree[TaskTrainerHistory],
    save_model_parameters: Array,
    best_save_idx: PyTree[Int[Array, "replicate"]],
    n_replicates: int,
    where_train: WhereFunc | dict[str, WhereFunc],
) -> PyTree[eqx.Module, 'T']:
    """Serializes models with the best parameters for each replicate and training condition."""
    # Get a function that returns the `where_func` used on a given iteration
    if isinstance(where_train, dict):
        where_train_idxs = _get_most_recent_idxs(
            [int(k) for k in where_train.keys()],
            model_record.n_batches,        
        )
        get_where_train = lambda idx: where_train[str(where_train_idxs[idx])]
    else:
        get_where_train = lambda idx: where_train
    
    # TODO: If any model parameters were trainable at the end of the training run, 
    # but were not trainable at the time of the best iteration, then this will keep 
    # the final parameters. However, we should probably keep the value of these parameters
    # at the best iteration, even though they were not trainable then, since perhaps they 
    # became trainable later (i.e. resulting in the final values) and this may have 
    # affected the final loss.
    # TODO: Similarly, I think this might fail if two replicates differ in whether a parameter 
    # was trainable at the best iteration; we'll end up trying to do a `tree_stack` on pytrees
    # where some array leaves are sometimes `None`. The solution to this is the same as the 
    # solution above: we need to select the best version of parameters that were trainable 
    # *at any point*
    best_saved_parameters = tree_stack([
        # Select the best parameters for each replicate, for all train histories
        jt.map(
            lambda train_history, best_idxs: tree_take_multi(
                # Filter out the parameters that were not trainable at the best iteration
                eqx.filter(
                    train_history.model_parameters, 
                    filter_spec_leaves(
                        train_history.model_parameters, 
                        get_where_train(save_model_parameters[int(best_idxs[i])]),
                    ),
                    is_leaf=is_module,
                ),
                [int(best_idxs[i]), i], 
                [0, 1],
            ),
            train_histories, best_save_idx,
            is_leaf=is_module,
        )
        for i in range(n_replicates)
    ])
    
    models_with_best_parameters = eqx.combine(models, best_saved_parameters)
    
    return models_with_best_parameters


def get_replicate_distribution_figure(
    measure: TrainStdDict[float, Shaped[Array, 'replicates']], 
    yaxis_title="",
) -> go.Figure:
    
    n_replicates = len(jt.leaves(measure)[0])
    
    df = pd.DataFrame(measure).reset_index().melt(id_vars='index')
    df["index"] = df["index"].astype(str)

    fig = go.Figure()

    strips = px.scatter(
        df,
        x='variable',
        y='value',
        color="index",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        # stripmode='overlay',
    )
    
    strips.update_traces(
        marker_size=10,
        marker_symbol='circle-open',
        marker_line_width=3,
    )

    violins = [
        go.Violin(
            x=[disturbance_std] * n_replicates,
            y=data,
            # box_visible=True,
            line_color='black',
            meanline_visible=True,
            fillcolor='lightgrey',
            opacity=0.6,
            name=f"{disturbance_std}",
            showlegend=False,   
            spanmode='hard',  
        )
        for disturbance_std, data in measure.items()
    ]
    
    fig.add_traces(violins)
    fig.add_traces(strips.data)

    fig.update_layout(
        xaxis_type='category',
        width=800,
        height=500,
        xaxis_title="Train disturbance std.",
        yaxis_title=yaxis_title,
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


def get_train_history_figures(
    histories: PyTree[TaskTrainerHistory, 'T'],
    best_saved_iteration_by_replicate,
) -> PyTree[go.Figure, 'T']:
    def get_figure(history, best_save_iterations):
        fig = fbp.loss_history(history.loss)
        text = "Best iter. by replicate: " + ", ".join(
            str(idx) for idx in best_save_iterations
        )
        fig.add_annotation(dict(
            text=text,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5, 
            y=1,
        ))
        return fig
        
    return jt.map(
        lambda history, best_save_iterations: get_figure(history, best_save_iterations),
        histories, best_saved_iteration_by_replicate,
        is_leaf=is_type(TaskTrainerHistory),
    )


class FigFuncSpec(eqx.Module):
    func: Callable[..., go.Figure | TrainStdDict[float, go.Figure]]
    args: tuple[PyTree, ...]


def save_training_figures(
    db_session,
    eval_info,
    train_histories, 
    replicate_info,
):
    # Specify the figure-generating functions and their arguments
    fig_specs: dict[str, FigFuncSpec] = dict(
        loss_history=FigFuncSpec(
            func=get_train_history_figures,
            args=(train_histories, replicate_info['best_saved_iteration_by_replicate']),
        ),
        loss_dist_over_replicates_best=FigFuncSpec(
            func=partial(
                get_replicate_distribution_figure, 
                yaxis_title=f"Best batch total loss",
            ),
            args=(replicate_info['losses_at_best_saved_iteration'],),
        ),
        loss_dist_over_replicates_final=FigFuncSpec(
            func=partial(
                get_replicate_distribution_figure, 
                yaxis_title=f"Final batch total loss",
            ),
            args=(replicate_info['losses_at_final_saved_iteration'],),
        ),
        readout_norm=FigFuncSpec(
            func=partial(
                get_replicate_distribution_figure, 
                yaxis_title=f"Frobenius norm of readout weights",
            ),
            args=(replicate_info['readout_norm'],), 
        ),
    )
    
    # Evaluate all of them at the TrainStdDict level
    all_figs = {
        fig_label: jt.map(
            fig_spec.func,
            *fig_spec.args,
            is_leaf=is_type(TrainStdDict),
        )
        for fig_label, fig_spec in fig_specs.items()
    }


    def save_and_add_figure(fig, plot_id, variant_label, train_std):
        fig_parameters = dict()
        
        if variant_label:
            fig_parameters |= dict(variant_label=variant_label)
        
        if train_std:
            fig_parameters |= dict(disturbance_train_std=float(train_std))
        
        add_evaluation_figure(
            db_session,
            eval_info,
            fig,
            plot_id,
            # TODO: let the user specify which formats to save
            save_formats=['png'],
            **fig_parameters,
        )

    # Save and add records for each figure
    for plot_id, figs in all_figs.items():
        # Some training notebooks use multiple training methods, and some don't. And some figure functions
        # return one figure per training condition, while others are summaries. Thus we need to descend 
        # to the `TrainStdDict` or `go.Figure` level first, and whatever the label is down to that level, will
        # label the training method (variant). Then we can descend to the `go.Figure` level, and whatever 
        # label is constructed here will either be the training std (if we originally descended to `TrainStdDict`),
        # or nothing.
        jt.map(
            # Map over each set (i.e. training variant) of disturbance train stds
            lambda fig_set, variant_label: jt.map(
                lambda fig, train_std: save_and_add_figure(fig, plot_id, variant_label, train_std),
                fig_set, tree_labels(fig_set, join_with="_", is_leaf=is_type(go.Figure)),
                is_leaf=is_type(go.Figure),
            ),
            figs,
            tree_labels(figs, join_with="_", is_leaf=is_type(TrainStdDict, go.Figure)),
            is_leaf=is_type(TrainStdDict, go.Figure),
        )

        logger.info(f"Saved {plot_id} figure set")


def compute_replicate_info(
    model_record,
    models,
    tasks,
    train_histories, 
    save_model_parameters, 
    n_replicates, 
    n_std_exclude,
    where_train,
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
    
    # Create models with best parameters
    best_models = get_best_models(
        model_record,
        models, 
        train_histories, 
        save_model_parameters,
        best_save_idx, 
        n_replicates, 
        where_train,
    )
    
    readout_norm = jt.map(
        lambda model: jnp.linalg.norm(model.step.net.readout.weight, axis=(-2, -1), ord='fro'),
        best_models,
        is_leaf=is_module,        
    )
    
    replicate_info = dict(
        best_save_idx=best_save_idx,
        best_saved_iteration_by_replicate=best_saved_iterations,
        losses_at_best_saved_iteration=losses_at_best_saved_iteration,
        losses_at_final_saved_iteration=losses_at_final_saved_iteration,
        best_replicates=best_replicates,
        included_replicates=included_replicates,
        readout_norm=readout_norm,
    )   
    
    return replicate_info, best_models
    
    
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
    process_all: bool = True,
) -> None:
    """Process a single model record, adding a new record with best parameters and replicate info."""
    
    if model_record.has_replicate_info or (model_record.postprocessed and not process_all):
        logger.info(f"Model {model_record.hash} has been processed previously and process_all is false; skipping")
        return
    
    origin = str(model_record.origin)
    # TODO: Either ignore the typing here, or make these columns explicit in `ModelRecord`
    where_train = jt.map(
        attr_str_tree_to_where_func,
        model_record.where_train_strs,
        is_leaf=is_type(list),
    )
    # where_train = attr_str_tree_to_where_func(tuple(set(jt.leaves(model_record.where_train_strs))))
    disturbance_type = str(model_record.disturbance_type)
    n_replicates = int(model_record.n_replicates)       
    save_model_parameters = jnp.array(model_record.save_model_parameters)
    
    all_data = load_data(model_record)
    
    if all_data is None:
        return
    else:
        models, model_hyperparams, train_histories, train_history_hyperparams = all_data
    
    # Get respective validation tasks for each model
    tasks = setup_tasks_only(
        SETUP_FUNCS[origin], 
        key=jr.PRNGKey(0), 
        **model_hyperparams,
    )
    
    # Compute replicate info``
    replicate_info, best_models = compute_replicate_info(
        model_record,
        models,
        tasks,
        train_histories, 
        save_model_parameters, 
        n_replicates, 
        n_std_exclude, 
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
            str(model_record.origin),
            best_models,
            model_hyperparams,
            record_hyperparameters | dict(n_std_exclude=n_std_exclude),
            train_history=train_histories,
            train_history_hyperparameters=train_history_hyperparams,
            replicate_info=replicate_info,
            replicate_info_hyperparameters=dict(n_replicates=n_replicates),
        )
        
    except Exception as e:
        # If anything fails, rollback and restore original record
        session.rollback()
        logger.error(f"Failed to process model {model_record.hash}: {e}")
        raise 
    
    #? Do we really need to make one of these, here? Or should training figures have their own table? 
    eval_info = add_evaluation(
        session,
        model_hash=model_record.hash,
        eval_parameters=dict(
            n_evals=N_TRIALS_VAL,
            # n_std_exclude=n_std_exclude,  # Not relevant to the figures that are generated?
        ),
        origin="post_training",
    )
    
    # Save training figures
    save_training_figures(
        session,
        eval_info,
        train_histories, 
        replicate_info,
    )
    
    model_record.postprocessed = int(True)
    session.commit()
    logger.info(f"Processed model {model_record.hash}")
    
    
def main(n_std_exclude: float = 2.0, process_all: bool = False):
    """Process all models in database."""
    session = get_db_session()
    
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
                    process_all,
                )
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Skipping model {model_record.hash} due to error: {e}")
                raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-training processing of models.")
    parser.add_argument("--n_std_exclude", default=2, type=float, help="Mark replicates this many stds above the best as to-be-excluded")
    parser.add_argument("--process_all", action="store_true", help="Reprocess all models, even if they already have replicate info")
    args = parser.parse_args()
    
    main(args.n_std_exclude, args.process_all)