from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import List, Optional

import dill as pickle
import equinox as eqx
import jax
import jax.tree as jt
from jaxtyping import PyTree
import optax
import plotly
import plotly.graph_objects as go
from sqlalchemy.orm import Session

import feedbax
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies.analysis import modules as analysis_modules_pkg
from rnns_learn_robust_motor_policies.analysis._dependencies import compute_dependency_results
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, get_validation_trial_specs, logger
from rnns_learn_robust_motor_policies.colors import COMMON_COLOR_SPECS, setup_colors
from rnns_learn_robust_motor_policies.config import PATHS
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.database import EvaluationRecord, ModelRecord, add_evaluation, check_model_files, get_db_session
from rnns_learn_robust_motor_policies.hyperparams import flatten_hps, load_hps, use_train_hps_when_none
from rnns_learn_robust_motor_policies.misc import delete_all_files_in_dir, log_version_info, load_module_from_package
from rnns_learn_robust_motor_policies.setup_utils import query_and_load_model
import rnns_learn_robust_motor_policies.training.modules as training_modules_pkg
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace, namespace_to_dict


def load_trained_models_and_aux_objects(training_module_name: str, hps: TreeNamespace, db_session: Session):
    """Given the analysis config, load the trained models and related objects (e.g. train tasks)."""
    
    training_module = load_module_from_package(training_module_name, training_modules_pkg)
    
    pairs, model_info, replicate_info, n_replicates_included = jtree.unzip(
        #? Should this structure be hardcoded here?
        #? At least for this project, we typically load spreads of trained models, 
        #? and those spreads are always over the training perturbation std.
        LDict.of("train__pert__std")({
            train_pert_std: query_and_load_model(
                db_session,
                training_module.setup_task_model_pair,
                params_query=namespace_to_dict(flatten_hps(hps.train)) | dict(
                    expt_name=training_module_name,
                    pert__std=train_pert_std
                ),
                noise_stds=dict(
                    feedback=hps.model.feedback_noise_std,
                    motor=hps.model.motor_noise_std,
                ),
                surgeries={
                    # Change
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


def setup_eval_for_module(
    analysis_name: str,
    hps: TreeNamespace, 
    db_session: Session,    
):
    """Given the analysis module, set up the evaluation(s).
    
    1. Construct the task-model pairs to evaluate, to produce the state needed for the analyses.
    2. Add an evaluation record to the database. 
    """
    
    analysis_module: ModuleType = load_module_from_package(analysis_name, analysis_modules_pkg)
    
    #! For this project, assume only a single-level mapping between training experiments and analysis subpackages;
    #! thus `analysis.modules.part1` corresponds to `training.modules.part1`.
    #! In general, it might be better to go back to explicitly specifying the `expt_name` for the 
    #! training experiment, in each analysis config file.
    training_module_name = analysis_name.split('.')[0]
    
    models_base, model_info, replicate_info, tasks_train, n_replicates_included = \
        load_trained_models_and_aux_objects(training_module_name, hps, db_session)

    #! TODO: Use the hyperparameters of the loaded model(s), where they were absent from the load spec
    #! (This should probably be moved to the model loading function)
    # model_hps = jt.map(record_to_namespace, model_info, is_leaf=is_type(RecordBase))
    # hps = jt.map(lambda hps, model_hps: model_hps | hps, hps, model_hps)

    #! For this project, the training task should not vary with the train field std 
    #! so we just keep a single one of them.
    # TODO: In the future, could keep the full `tasks_base`, and update `get_task_variant`/`setup_func`
    task_base = jt.leaves(tasks_train, is_leaf=is_module)[0]

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
        expt_name=analysis_name,
        models=model_info,
        #? Could move the flattening/conversion to `database`?
        eval_parameters=namespace_to_dict(flatten_hps(hps)),
        version_info=version_info,
    )

    def get_task_variant(task_base, models_base, hps, variant_key, **kwargs):
        task = task_base

        for attr_name, attr_value in kwargs.items():
            task = eqx.tree_at(
                lambda task: getattr(task, attr_name),
                task,
                attr_value,
            )

        tasks, models, hps, extras = analysis_module.setup_eval_tasks_and_models(
            task, models_base, hps
        )

        hps = jt.map(
            lambda hps: eqx.tree_at(
                lambda hps: hps.task,
                hps,
                getattr(hps.task, variant_key),
            ),
            hps,
            is_leaf=is_type(TreeNamespace),
        )

        return tasks, models, hps, extras

    # Outer level is task variants, inner is the structure returned by `setup_func`
    # i.e. "task variants" are a way to evaluate different sets of conditions
    all_tasks, all_models, all_hps, all_extras = jtree.unzip({
        variant_key: get_task_variant(
            task_base,
            models_base,
            hps,
            variant_key,
            n_steps=hps.model.n_steps,  #? Is this the only one we need to pass explicitly?
            **variant_params,
        )
        for variant_key, variant_params in namespace_to_dict(hps.task).items()
    })

    return (
        analysis_module,
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
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    model_info: ModelRecord,
    eval_info: EvaluationRecord,
    *,
    fig_dump_path: Optional[Path] = None,
    fig_dump_formats: List[str] = ["html"],
    custom_dependencies: Optional[dict[str, AbstractAnalysis]] = None,
    **kwargs,
) -> tuple[PyTree, PyTree[go.Figure]]:
    """Given a list or dict of instances of `AbstractAnalysis`, perform all analyses and save any figures."""
    
    # Phase 1: Compute all analysis nodes (dependencies + leaves) 
    all_dependency_results = compute_dependency_results(analyses, data, custom_dependencies, **kwargs)

    if not any(analyses):
        raise ValueError("No analyses given to perform")

    # Phase 2: Generate figures for leaf analyses only
    def make_figs_and_save(analysis_key: str, analysis: AbstractAnalysis, dependencies: dict):
        logger.info(f"Making figures: {analysis_key}")
        # Get the computed result for this analysis (computed in phase 1)
        result = dependencies.pop('result')
        
        figs = analysis._make_figs_with_ops(data, result, **dependencies)
        
        if figs is not None:
            analysis.save_figs(
                db_session,
                eval_info,
                result,
                figs,
                data.hps,
                model_info,
                dump_path=fig_dump_path,
                dump_formats=fig_dump_formats,
                **dependencies,
            )
        logger.info(f"Figures saved: {analysis_key}")
        return analysis, result, figs

    all_analyses, all_results, all_figs = jtree.unzip({
        analysis_key: make_figs_and_save(analysis_key, analysis, dependencies)
        for (analysis_key, analysis), dependencies in zip(analyses.items(), all_dependency_results)
    })

    return all_analyses, all_results, all_figs


def run_analysis_module(
    analysis_name: str,
    fig_dump_path: Optional[str] = None,
    fig_dump_formats: List[str] = ["html", "webp", "svg"],
    no_pickle: bool = False,
    retain_past_fig_dumps: bool = False,
    states_pkl_dir: Optional[Path] = PATHS.states_tmp,
    *,
    key,
):
    """Given the path/string id of an analysis module, run it.
    
    1. Construct all task-model pairs defined by the module's `setup_eval_tasks_and_models` 
       function, then evaluate them according to the module's `eval_func`. The result is 
       a PyTree of states for different evaluation/training conditions. 
    2. Perform all analyses defined by the module's `ALL_ANALYSES` attribute,
       given all the available data (states, tasks, models, hyperparameters, etc.).
    """
    if fig_dump_path is None:
        fig_dump_path = PATHS.figures_dump

    # Start a database session for loading trained models, and saving evaluation/figure records
    db_session = get_db_session()
    check_model_files(db_session)  # Ensure we don't try to load any models whose files don't exist

    # Load the config (hyperparameters) for the analysis module
    hps = load_hps(analysis_name, config_type='analysis')
    # If some config values (other than those under the `load` key) are unspecified, replace them with 
    # respective values from the `load` key
    # e.g. if trained on curl fields and hps.pert.type is None, use hps.train.pert.type
    # (In `setup_tasks_and_models` we then fill out any fields that are entirely missing from the config, 
    #  but which are given by the loaded model records.)
    hps_common = use_train_hps_when_none(hps)

    analysis_module, tasks, models, hps, extras, model_info, eval_info, replicate_info = \
        setup_eval_for_module(
            analysis_name,
            hps_common,
            db_session,
        )

    trial_specs = jt.map(get_validation_trial_specs, tasks, is_leaf=is_module)

    #! Currently, colorscales do not vary with the task variant or conditions, but are just determined by the 
    #! spreads of hyperparameters. I don't think there's been a case yet where we've constructed spreads of 
    #! hyperparameters (rather than single values) that differ between variants/conditions. 
    # TODO: In case we need to color by e.g. reach directio
    colors, colorscales = setup_colors(hps_common, COMMON_COLOR_SPECS | analysis_module.COLOR_FUNCS)
    # colors_0 = {
    #     variant_label: jt.leaves(variant_hps, is_leaf=LDict.is_of("color"))[0]
    #     for variant_label, variant_hps in colors.items()
    # }

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

    # Helper function to compute states from scratch
    def _compute_states_and_log_memory_estimate():
        states_shapes = eqx.filter_eval_shape(
            evaluate_all_states, tasks, models, hps
        )
        logger.info(f"{jtree.struct_bytes(states_shapes) / 1e9:.2f} GB of memory estimated to store all states.")

        computed_states = evaluate_all_states(tasks, models, hps)
        logger.info("All states evaluated.")
        return computed_states

    # Create a filename based on the evaluation hash
    states_pickle_path = states_pkl_dir / f"{eval_info.hash}.pkl"

    loaded_from_pickle = False
    # If --no-pickle is set, we won't try to load from or save to pickle
    if not no_pickle and states_pickle_path.exists():
        # Try to load from pickle
        logger.info(f"Loading states from {states_pickle_path}...")
        try:
            with open(states_pickle_path, 'rb') as f:
                states = pickle.load(f)
            logger.info("States loaded from pickle.")
            loaded_from_pickle = True
        except Exception as e:
            logger.error(f"Failed to load pickled states: {e}")
            logger.info("Computing states from scratch instead...")
            states = _compute_states_and_log_memory_estimate()
    else:
        if no_pickle and states_pickle_path.exists():
            logger.info(f"Ignoring pickle file at {states_pickle_path} due to --no-pickle flag.")

        # Compute from scratch
        states = _compute_states_and_log_memory_estimate()

    # Save states if we didn't use --no-pickle and we didn't successfully load from pickle
    if not no_pickle and not loaded_from_pickle:
        def _test(tree):
            return jt.map(lambda x: x, tree)

        with open(states_pickle_path, 'wb') as f:
            pickle.dump(_test(states), f)
        logger.info(f"Saved evaluated states to {states_pickle_path}")

    # ANY subclass of `AbstractAnalysis` can add any of the following to the argument lists of
    # their `make_figs` and `compute` methods
    common_inputs = dict(
        hps_common=hps_common,
        colors=colors,
        colorscales=colorscales,
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
    
    if not retain_past_fig_dumps:
        try:
            delete_all_files_in_dir(fig_dump_path)
            logger.info(f"Deleted existing dump figures in {fig_dump_path}")
        except ValueError as e:
            logger.warning(f"Failed to delete existing dump figures: {e}; directory probably doesn't exist yet")

    all_analyses, all_results, all_figs = perform_all_analyses(
        db_session,
        analysis_module.ALL_ANALYSES,
        data,
        model_info,
        eval_info,
        fig_dump_path=Path(fig_dump_path),
        fig_dump_formats=fig_dump_formats,
        custom_dependencies=getattr(analysis_module, 'DEPENDENCIES', {}),
        **common_inputs,
    )

    return data, common_inputs, all_analyses, all_results, all_figs