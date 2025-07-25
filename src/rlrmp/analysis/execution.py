from collections.abc import Callable, Sequence
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Union

import dill as pickle
import equinox as eqx
from equinox import Module
import jax
import jax.tree as jt
from jaxtyping import PyTree
import optax
import plotly
import plotly.graph_objects as go
from sklearn import tree
from sqlalchemy.orm import Session

import feedbax
from jax_cookbook import is_module, is_type, is_none
import jax_cookbook.tree as jtree

import rlrmp
from rlrmp.analysis import modules as analysis_modules_pkg
from rlrmp.analysis._dependencies import compute_dependency_results
from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisInputData, get_validation_trial_specs, logger
from rlrmp.colors import COMMON_COLOR_SPECS, setup_colors
# Access project paths and string constants
from rlrmp.config import PATHS
from rlrmp.constants import REPLICATE_CRITERION
# `record_to_dict` converts SQLAlchemy records to plain dicts
from rlrmp.database import (
    EvaluationRecord,
    ModelRecord,
    add_evaluation,
    check_model_files,
    get_db_session,
)
# Added utilities for unflattening record hyperparameters into namespaces  
from rlrmp.database import fill_missing_train_hps_from_record, fill_hps_with_train_params
from rlrmp.tree_utils import tree_level_labels
from rlrmp.types import (
    LDict,
    TreeNamespace,
    namespace_to_dict,
)
# `cast_hps` is needed to convert dictionaries (e.g. `where`) back into the expected objects
from rlrmp.hyperparams import (
    flatten_hps,
    load_hps,
    use_train_hps_when_none,
)
from rlrmp.misc import delete_all_files_in_dir, log_version_info, load_module_from_package
from rlrmp.setup_utils import query_and_load_model
import rlrmp.training.modules as training_modules_pkg


STATES_CACHE_SUBDIR = "states"


# === Transform System ===

# Pre-setup can be either granular dict or combined function
PreSetupSpec = Union[
    dict[str, Optional[Callable]],  # Runtime validation will check keys are 'task'/'models'
    Callable[[Module, LDict[float, Module]], tuple[Module, LDict[float, Module]]]
]


class AnalysisModuleTransformSpec(Module):
    """Specifies transformations to apply at different stages of analysis execution.
    
    Attributes
    ----------
    pre_setup : Optional[PreSetupSpec]
        Transform applied to task_base and models_base before setup_eval_tasks_and_models.
        Can be either:
        - dict with 'task' and/or 'models' keys mapping to transform functions
        - single function taking (task, models) and returning (task, models)
        Useful for operations like selecting best replicates to save computation.
    post_eval : Optional[Callable[[PyTree], PyTree]]
        Transform applied to evaluation results (data.states) after evaluation completes.
        Useful for post-processing results while preserving intermediate data.
    """
    pre_setup: Optional[PreSetupSpec] = None
    post_eval: Optional[Callable[[PyTree], PyTree]] = None


# Also allow plain dict for convenience
TransformsSpec = Union[
    AnalysisModuleTransformSpec,  # Structured (better static typing)
    dict[str, Any]  # Dict (convenience) - runtime validation will check keys
]


def _call_user_func(func, data, extra_kwargs):
    """Invoke func with data and the subset of extra_kwargs it accepts.
    
    This allows transform functions to optionally accept common inputs like replicate_info.
    """
    sig = inspect.signature(func)
    
    # Fast path: the function accepts **kwargs; forward everything.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(data, **extra_kwargs)
    
    # Otherwise filter down to only the kwargs explicitly listed.
    filtered = {
        k: v for k, v in extra_kwargs.items() 
        if k in sig.parameters
    }
    return func(data, **filtered)


def validate_and_convert_transforms(transforms_spec: Optional[TransformsSpec]) -> AnalysisModuleTransformSpec:
    """Convert and validate transforms specification to canonical form."""
    if transforms_spec is None:
        return AnalysisModuleTransformSpec()
        
    if isinstance(transforms_spec, dict):
        # Validate top-level keys
        valid_keys = {'pre_setup', 'post_eval'}
        invalid_keys = set(transforms_spec.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid transform keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )
        
        pre_setup = transforms_spec.get('pre_setup')
        post_eval = transforms_spec.get('post_eval')
        
        # Validate pre_setup if present
        if pre_setup is not None and isinstance(pre_setup, dict):
            valid_pre_keys = {'task', 'models'}
            invalid_pre_keys = set(pre_setup.keys()) - valid_pre_keys
            if invalid_pre_keys:
                raise ValueError(
                    f"Invalid pre_setup keys: {invalid_pre_keys}. "
                    f"Valid keys are: {valid_pre_keys}"
                )
        
        return AnalysisModuleTransformSpec(
            pre_setup=pre_setup,
            post_eval=post_eval
        )
    
    elif isinstance(transforms_spec, AnalysisModuleTransformSpec):
        return transforms_spec
    
    else:
        raise ValueError(
            f"transforms_spec must be dict or AnalysisModuleTransformSpec, "
            f"got {type(transforms_spec)}"
        )


def load_trained_models_and_aux_objects(training_module_name: str, hps: TreeNamespace, db_session: Session):
    """Given the analysis config, load the trained models and related objects (e.g. train tasks)."""
    
    training_module = load_module_from_package(training_module_name, training_modules_pkg)
    
    pairs, model_info, replicate_info, n_replicates_included, hps_train_dict = jtree.unzip(
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

    return models, model_info, replicate_info, tasks_train, n_replicates_included, hps_train_dict


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
    
    models_base, model_info, replicate_info, tasks_train, n_replicates_included, hps_train_dict = \
        load_trained_models_and_aux_objects(training_module_name, hps, db_session)

    # Fill-in any missing training hyper-parameters **out-of-place** and switch to
    # the enriched version from here onward. Use training hps from database records.
    # Take the first entry from hps_train_dict as representative training hyperparameters
    hps_train_representative = jt.leaves(hps_train_dict, is_leaf=is_type(TreeNamespace))[0]
    hps_filled = fill_hps_with_train_params(hps, hps_train_representative)
    hps = hps_filled  #  use the enriched version for everything below

    #! For this project, the training task should not vary with the train field std 
    #! so we just keep a single one of them.
    # TODO: In the future, could keep the full `tasks_base`, and update `get_task_variant`/`setup_func`
    task_base = jt.leaves(tasks_train, is_leaf=is_module)[0]

    # Load and validate transforms once
    transforms_raw = getattr(analysis_module, 'TRANSFORMS', None)
    transforms = validate_and_convert_transforms(transforms_raw)

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
        jax, eqx, optax, plotly, git_modules=(feedbax, rlrmp),
    )

    # Add evaluation record to the database
    eval_info = add_evaluation(
        db_session,
        expt_name=analysis_name,
        models=model_info,
        #? Could move the flattening/conversion to `database`?
        #! TODO: Could exclude train parameters, since 
        eval_parameters=namespace_to_dict(flatten_hps(hps)),
        version_info=version_info,
    )

    #? Should this be a method of `AbstractTask`?
    def _get_task_variant(task: Module, variant_params: dict[str, Any]) -> Module:
        """Get a task variant based on the base task and the variant parameters."""
        for attr_name, attr_value in variant_params.items():
            #! TODO: Might be quicker to do a single `tree_at` with a tuple-of-attrs accessor and a
            #! tuple of values
            task = eqx.tree_at(
                lambda task: getattr(task, attr_name),
                task_base,
                attr_value,
            )
        return task

    # Construct common inputs needed by transforms and analyses
    # Note: We construct trial_specs for all task variants here
    task_variants = LDict.of('task_variant')({
        variant_key: _get_task_variant(task_base, variant_params)
        for variant_key, variant_params in namespace_to_dict(hps.task).items()
    })
    
    trial_specs = jt.map(get_validation_trial_specs, task_variants, is_leaf=is_module)
    
    colors, colorscales = setup_colors(hps, COMMON_COLOR_SPECS | analysis_module.COLOR_FUNCS)
    
    common_inputs = dict(
        hps_common=hps,
        colors=colors,
        colorscales=colorscales,
        replicate_info=replicate_info,
        trial_specs=trial_specs,
    )

    def setup_per_task_variant(task, models_base, hps, variant_key, **kwargs):
        # Apply pre-setup transformations if present
        if transforms.pre_setup is not None:
            if isinstance(transforms.pre_setup, dict):
                # Granular transforms - apply to task and/or models separately
                if 'task' in transforms.pre_setup and transforms.pre_setup['task'] is not None:
                    task = _call_user_func(transforms.pre_setup['task'], task, common_inputs)
                if 'models' in transforms.pre_setup and transforms.pre_setup['models'] is not None:
                    models_base = _call_user_func(transforms.pre_setup['models'], models_base, common_inputs)
            else:
                # Combined function - pass task and models as tuple
                task, models_base = _call_user_func(transforms.pre_setup, (task, models_base), common_inputs)

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
    all_tasks, all_models, all_hps, all_extras = jtree.unzip(LDict.of('task_variant')({
        variant_key: setup_per_task_variant(
            task_variant, 
            models_base, 
            hps, 
            variant_key,
        )
        for variant_key, task_variant in task_variants.items()
    }))
    
    data = AnalysisInputData(
        hps=all_hps,
        tasks=all_tasks,
        models=all_models,
        states=None,  # Empty prior to evaluation
        extras=all_extras,
    )

    return (
        analysis_module,
        data,
        common_inputs,  # includes replicate_info, trial_specs, colors, etc.
        transforms,  # validated transforms
        model_info,
        eval_info,
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
) -> tuple[PyTree[AbstractAnalysis], PyTree[Any], PyTree[go.Figure]]:
    """Given a list or dict of instances of `AbstractAnalysis`, perform all analyses and save any figures."""
    
    if not analyses:
        logger.warning("No analyses given to perform; nothing returned")
        return None, None, None
    
    if not all(isinstance(value, AbstractAnalysis) for value in analyses.values()):
        raise ValueError("All analyses defined in given analysis module must be instances of `AbstractAnalysis`")
    
    # Phase 1: Compute all analysis nodes (dependencies + leaves) 
    all_dependency_results = compute_dependency_results(analyses, data, custom_dependencies, **kwargs)

    if not any(analyses):
        raise ValueError("No analyses given to perform")

    # Phase 2: Keep results & generate figures for leaf analyses only
    def finish_analysis(analysis_key: str, analysis: AbstractAnalysis, inputs: dict):
        logger.info(f"Making figures: {analysis_key}")
        # Get the computed result for this analysis (computed in phase 1)
        result = inputs.pop('result')
        
        figs = analysis._make_figs_with_ops(data, result, **inputs)
        
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
                label=analysis_key,
                **inputs,
            )
            logger.info(f"Figures saved: {analysis_key}")
            
        return analysis, result, figs

    all_analyses, all_results, all_figs = jtree.unzip({
        analysis_key: finish_analysis(analysis_key, analysis, dependencies)
        for (analysis_key, analysis), dependencies in zip(analyses.items(), all_dependency_results)
    })

    return all_analyses, all_results, all_figs


def run_analysis_module(
    analysis_name: str,
    fig_dump_path: Optional[str] = None,
    fig_dump_formats: List[str] = ["html", "webp", "svg"],
    no_pickle: bool = False,
    retain_past_fig_dumps: bool = False,
    states_pkl_dir: Optional[Path] = PATHS.cache / "states",
    eval_only: bool = False,  # Skip analyses and just evaluate the states
    *,
    key,
):
    """Given the path/string id of an analysis module, run it.
    
    1. Construct all task-model pairs defined by the module's `setup_eval_tasks_and_models` 
       function, then evaluate them according to the module's `eval_func`. The result is 
       a PyTree of states for different evaluation/training conditions. 
    2. Perform all analyses defined by the module's `ANALYSES` attribute,
       given all the available data (states, tasks, models, hyperparameters, etc.).
    """
    if fig_dump_path is None:
        fig_dump_path = PATHS.figures_dump

    # Ensure the directory for state pickles exists
    if states_pkl_dir is None:
        states_pkl_dir = PATHS.cache / STATES_CACHE_SUBDIR
    assert states_pkl_dir is not None
    states_pkl_dir.mkdir(parents=True, exist_ok=True)

    # Start a database session for loading trained models, and saving evaluation/figure records
    db_session = get_db_session()
    check_model_files(db_session)  # Ensure we don't try to load any models whose files don't exist

    # Load the config (hyperparameters) for the analysis module
    hps = load_hps(analysis_name, config_type='analysis')
    # Establish a provisional common namespace (needed for querying the DB); this
    # will be superseded after we load the model records and fill in any missing
    # hyper-parameters.
    provisional_hps_common = use_train_hps_when_none(hps)

    (
        analysis_module, 
        data,
        common_inputs, 
        transforms,
        model_info, 
        eval_info, 
    ) = \
        setup_eval_for_module(
            analysis_name,
            provisional_hps_common,
            db_session,
        )


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
            evaluate_all_states, data.tasks, data.models, data.hps
        )
        logger.info(f"{jtree.struct_bytes(states_shapes) / 1e9:.2f} GB of memory estimated to store all states.")

        computed_states = evaluate_all_states(data.tasks, data.models, data.hps)
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
            logger.info(f"Loaded pickled states with PyTree structure: {tree_level_labels(states, is_leaf=is_module)}")
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
        logger.info(f"Computed states with PyTree structure: {tree_level_labels(states, is_leaf=is_module)}")

    # Save states if we didn't use --no-pickle and we didn't successfully load from pickle
    if not no_pickle and not loaded_from_pickle:
        def _test(tree):
            return jt.map(lambda x: x, tree)

        with open(states_pickle_path, 'wb') as f:
            pickle.dump(_test(states), f)
        logger.info(f"Saved evaluated states to {states_pickle_path}")

    # Apply post-eval transformations if present
    if transforms.post_eval is not None:
        states = _call_user_func(transforms.post_eval, states, common_inputs)

    data = eqx.tree_at(lambda data: data.states, data, states, is_leaf=is_none)

    if eval_only:
        logger.info("Eval-only requested; skipping analyses and returning.")
        return data, common_inputs, None, None, None
    
    if not retain_past_fig_dumps:
        try:
            delete_all_files_in_dir(Path(fig_dump_path))
            logger.info(f"Deleted existing dump figures in {fig_dump_path}")
        except ValueError as e:
            logger.warning(f"Failed to delete existing dump figures: {e}; directory probably doesn't exist yet")

    all_analyses, all_results, all_figs = perform_all_analyses(
        db_session,
        analysis_module.ANALYSES,
        data,
        model_info,
        eval_info,
        fig_dump_path=Path(fig_dump_path),
        fig_dump_formats=fig_dump_formats,
        custom_dependencies=getattr(analysis_module, 'DEPENDENCIES', {}),
        **common_inputs,
    )

    return data, common_inputs, all_analyses, all_results, all_figs
