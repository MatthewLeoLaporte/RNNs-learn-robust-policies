#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
import logging
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax 
import plotly
from sqlalchemy.orm import Session

import feedbax
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree  

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies.config import PATHS, PRNG_CONFIG
from rnns_learn_robust_motor_policies.analysis import ANALYSIS_REGISTRY
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData
from rnns_learn_robust_motor_policies.analysis._dependencies import compute_dependencies
from rnns_learn_robust_motor_policies.colors import setup_colors
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.database import (
    EvaluationRecord, 
    ModelRecord,
    RecordBase, 
    add_evaluation, 
    check_model_files, 
    get_db_session, 
    # record_to_namespace,
)
from rnns_learn_robust_motor_policies.hyperparams import flatten_hps, load_hps
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training.post_training import TRAINPAIR_SETUP_FUNCS
from rnns_learn_robust_motor_policies.setup_utils import query_and_load_model
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import namespace_to_dict
from rnns_learn_robust_motor_policies.types import LDict


logger = logging.getLogger(os.path.basename(__file__))


def load_model_and_training_task(db_session: Session, hps: TreeNamespace):
    setup_task_model_pair = TRAINPAIR_SETUP_FUNCS[int(hps.train.expt_id)]
    
    pairs, model_info, replicate_info, n_replicates_included = jtree.unzip(
        #? Should this structure be hardcoded here?
        LDict.of("train__pert__std")({
            train_pert_std: query_and_load_model(
                db_session,
                setup_task_model_pair,
                params_query=namespace_to_dict(flatten_hps(hps.train)) | dict(
                    pert__std=train_pert_std
                ),
                noise_stds=dict(
                    feedback=hps.model.feedback_noise_std,
                    motor=hps.model.motor_noise_std,
                ),
                surgeries={
                    ('n_steps',): hps.model.n_steps,
                },  
                exclude_underperformers_by=REPLICATE_CRITERION,
                return_task=True,
            )
            for train_pert_std in hps.train.pert.std
        })
    )
    
    tasks_train, models = jtree.unzip(pairs) 
    
    return models, model_info, replicate_info, tasks_train, n_replicates_included


def copy_delattr(obj: Any, *attr_names: str):
    """Return a deep copy of an object, with some attributes removed."""
    obj = deepcopy(obj)
    for attr_name in attr_names:
        delattr(obj, attr_name)
    return obj


def use_train_hps_when_none(hps: TreeNamespace) -> TreeNamespace:
    """Replace any unspecified evaluation params with matching loading (training) params"""
    hps_train = hps.train
    hps_other = copy_delattr(hps, 'train')
    hps = hps_other.update_none_leaves(hps_train)
    hps.train = hps_train  
    return hps


"""
Default colorscales to try to set up, based on hyperparameters.
Values are hyperparameter where-functions so we can try to load them one-by-one.
"""
COMMON_COLOR_FUNCS = dict(
    # context_input= 
    pert__amp=lambda hps: hps.pert.amp,
    train__pert__std=lambda hps: hps.train.pert.std,
    # pert_var=  #? Discrete
    #  reach_condition=  #? Discrete
    context_input=lambda hps: hps.context_input,
    trial=lambda hps: range(hps.eval_n),  #? Discrete
)


def main(
    db_session, 
    hps_common: TreeNamespace,
    fig_dump_path: Optional[str] = None,
    *,
    key,
):    
    # If some config values (other than those under the `load` key) are unspecified, replace them with 
    # respective values from the `load` key
    # e.g. if trained on curl fields and hps.pert.type is None, use hps.train.pert.type
    # (In `setup_tasks_and_models` we then fill out any fields that are entirely missing from the config, 
    #  but which are given by the loaded model records.)
    hps_common = use_train_hps_when_none(hps_common)

    analysis_module = ANALYSIS_REGISTRY[hps_common.expt_id]
    
    tasks, models, hps, extras, model_info, eval_info, replicate_info = \
        setup_tasks_and_models(
            hps_common, 
            analysis_module.setup_eval_tasks_and_models, 
            db_session,
        )
    
    def get_validation_trial_specs(task):
        # TODO: Support any number of extra axes (i.e. for analyses that vmap over multiple axes in their task/model objects)
        if len(task.workspace.shape) == 3:
            return eqx.filter_vmap(lambda task: task.validation_trials)(task)
        else:
            return task.validation_trials
            
    trial_specs = jt.map(get_validation_trial_specs, tasks, is_leaf=is_module)
    
    colors, discrete_colors = setup_colors(hps, COMMON_COLOR_FUNCS | analysis_module.COLOR_FUNCS) 
    colors_0 = {
        variant_label: jt.leaves(variant_hps, is_leaf=LDict.is_of("color"))[0]
        for variant_label, variant_hps in colors.items()
    }
    
    def evaluate_all_states(all_tasks, all_models, all_hps):
        return jt.map(  # Map over the task-base model subtree pairs generated by `schedule_intervenor` for each base task
            lambda task, models, hps: jt.map(  # Map over the base model subtree, for the given base task
                lambda model: analysis_module.eval_func(key, hps, model, task),
                models,
                is_leaf=is_module,
            ),
            all_tasks,
            all_models,    
            all_hps,          
            is_leaf=is_module,
        )

    states_bytes = jtree.struct_bytes(eqx.filter_eval_shape(
        evaluate_all_states, tasks, models, hps
    ))
    logger.info(f"{states_bytes / 1e9:.2f} GB of memory estimated to store all states.")
    
    states = evaluate_all_states(tasks, models, hps)
    
    logger.info("All states evaluated.")
    
    # ANY subclass of `AbstractAnalysis` can add any of the following to the argument lists of
    # their `make_figs` and `compute` methods
    common_inputs = dict(
        hps_common=hps_common, 
        colors=colors,  
        colors_0=colors_0,  # Colors, assuming they only vary with the task variant and not the subconditions
        discrete_colors=discrete_colors,
        replicate_info=replicate_info,
        trial_specs=trial_specs,
    )
    
    data = AnalysisInputData(
        hps=hps,
        tasks=tasks,
        models=models,
        states=states,
        extras=extras,
    )
    
    all_results, all_figs = perform_all_analyses(
        db_session,
        analysis_module.ALL_ANALYSES, 
        data,
        model_info, 
        eval_info, 
        fig_dump_path=Path(PATHS.figures_dump),
        **common_inputs,
    )
    
    return data, all_results, all_figs


def setup_tasks_and_models(hps: TreeNamespace, setup_func: Callable, db_session: Session):
    # Load models
    models_base, model_info, replicate_info, tasks_base, n_replicates_included = \
        load_model_and_training_task(db_session, hps)
    
    #! TODO: Use the hyperparameters of the loaded model(s), where they were absent from the load spec
    #! (This should probably be moved to the model loading function)
    # model_hps = jt.map(record_to_namespace, model_info, is_leaf=is_type(RecordBase))
    # hps = jt.map(lambda hps, model_hps: model_hps | hps, hps, model_hps)
        
    #! For this project, the training task should not vary with the train field std 
    #! so we just keep a single one of them.
    # TODO: In the future, could keep the full `tasks_base`, and update `get_task_variant`/`setup_func`
    task_base = jt.leaves(tasks_base, is_leaf=is_module)[0]
    
    # If there is no system noise (i.e. the stds are zero), set the number of evaluations per condition to zero.
    # (Is there any other reason than the noise samples, why evaluations might differ?)
    # TODO: Make this optional? 
    #? What is the point of using `jt.leaves` here? 
    any_system_noise = any(jt.leaves((
        hps.model.feedback_noise_std,
        hps.model.motor_noise_std,
    )))
    if not any_system_noise:
        hps.eval_n = 1

    # Get indices for taking important subsets of replicates
    # best_replicate, included_replicates = jtree.unzip(LDict.of("train__pert__std")({
    #     std: (
    #         replicate_info[std]['best_replicates'][REPLICATE_CRITERION],
    #         replicate_info[std]['included_replicates'][REPLICATE_CRITERION],
    #     ) 
    #     # Assumes that `train.pert.std` is given as a sequence
    #     for std in hps.train.pert.std
    # }))
    
    version_info = log_version_info(
        jax, eqx, optax, plotly, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    # Add evaluation record to the database
    eval_info = add_evaluation(
        db_session,
        expt_id=str(hps.expt_id),
        models=model_info,
        #? Could move the flattening/conversion to `database`?
        eval_parameters=namespace_to_dict(flatten_hps(hps)),
        version_info=version_info,
    )

    def get_task_variant(task_base, models_base, hps, **kwargs):
        task = task_base
        
        for attr_name, attr_value in kwargs.items():
            task = eqx.tree_at(
                lambda task: getattr(task, attr_name),
                task,
                attr_value,
            )
    
        tasks, models, hps, extras = setup_func(task, models_base, hps)
        
        return tasks, models, hps, extras
    
    # Outer level is task variants, inner is the structure returned by `setup_func`
    # i.e. "task variants" are a way to evaluate different sets of conditions
    all_tasks, all_models, all_hps, all_extras = jtree.unzip({
        k: get_task_variant(
            task_base, 
            models_base, 
            hps, 
            n_steps=hps.model.n_steps,  #? Is this the only one we need to pass explicitly?
            **task_params,
        )
        for k, task_params in namespace_to_dict(hps.task).items()
    })
    
    return (
        all_tasks, 
        all_models, 
        all_hps, 
        all_extras, 
        model_info, 
        eval_info, 
        replicate_info,
    )


def perform_all_analyses(
    db_session: Session, 
    analyses: Sequence[AbstractAnalysis], 
    data: AnalysisInputData,
    model_info: ModelRecord, 
    eval_info: EvaluationRecord, 
    *,
    fig_dump_path: Optional[Path] = None,
    **kwargs,
):
    # Each value in `analyses` is a function that is passed a bunch of information and returns some result.
    # e.g. the result of `"aligned_vars": get_aligned_vars` will be passed to *all* analyses
    # This ensures that dependencies are only calculated once.
    # However, note that dependencies should have unique keys since they will be aggregated into a 
    # single dict before performing the analyses.
    # TODO: Only pass the dependencies to each analysis that it actually needs
    dependencies = compute_dependencies(analyses, data, **kwargs)

    if not any(analyses):
        raise ValueError("No analyses given to perform")
    
    def analyse_and_save(analysis: AbstractAnalysis):
        # Get all figures and results for this analysis.
        # Pass *all* dependencies to every analysis!
        logger.info(f"Start analysis: {analysis.name}")
        result, figs = analysis(data, **dependencies)
        analysis.save_figs(
            db_session, 
            eval_info, 
            result, 
            figs, 
            data.hps, 
            model_info, 
            dump_path=fig_dump_path,
            **dependencies,
        )
        logger.info(f"Results saved: {analysis.name}")
        return result, figs
    
    all_results, all_figs = jtree.unzip(jt.map(
        analyse_and_save, 
        analyses, 
        is_leaf=is_type(AbstractAnalysis),
    ))
    
    return all_results, all_figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    # Because of the behaviour of `load_hps`, config_path can also be the `expt_id: str` (i.e. YAML 
    # filename relative to `../src/rnns_learn_robust_motor_policies/config`) of a default config to load. 
    # This assumes there is no file whose relative path is identical to that `expt_id`.
    parser.add_argument("config_path", type=str, help="Path to the config file, or `expt_id` of a default config.")
    parser.add_argument("--fig-dump-path", type=str, default="/tmp/fig_dump", help="Path to dump figures.")
    args = parser.parse_args()
    hps = load_hps(args.config_path, config_type='analysis')

    db_session = get_db_session()
    
    check_model_files(db_session)  # Mark any records with missing model files
    
    key = jr.PRNGKey(PRNG_CONFIG.seed)
    _, _, key_eval = jr.split(key, 3)
    
    main(db_session, hps, fig_dump_path=args.fig_dump_path, key=key)
    
    
    