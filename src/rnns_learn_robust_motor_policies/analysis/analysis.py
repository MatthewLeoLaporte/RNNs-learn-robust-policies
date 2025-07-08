from collections.abc import Callable, Hashable, Mapping, Sequence
import dataclasses
from functools import cached_property, partial, wraps
import inspect
import logging
from types import LambdaType, MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal, NamedTuple, Optional, Dict, Self, TypeAlias, Union
from pathlib import Path
import yaml

import equinox as eqx
from equinox import AbstractVar, AbstractClassVar, Module
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import ArrayLike, PyTree, Array
import plotly.graph_objects as go
from sqlalchemy.orm import Session

from feedbax.task import AbstractTask
from jax_cookbook import is_type, is_none, vmap_multi, natural_to_sequential_axes
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.config.config import STRINGS
from rnns_learn_robust_motor_policies.database import EvaluationRecord, add_evaluation_figure, savefig
from rnns_learn_robust_motor_policies.tree_utils import move_ldict_level_above, subdict, tree_level_labels, ldict_level_to_top
from rnns_learn_robust_motor_policies.misc import camel_to_snake, get_dataclass_fields, get_md5_hexdigest, get_name_of_callable, is_json_serializable
from rnns_learn_robust_motor_policies.plot_utils import figs_flatten_with_paths, get_label_str
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


logger = logging.getLogger(__name__)


# Define a string representer for objects PyYAML doesn't know how to handle
def represent_undefined(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
yaml.add_representer(object, represent_undefined)


class AnalysisInputData(Module):
    models: PyTree[Module]
    tasks: PyTree[Module]
    states: PyTree[Module]
    hps: PyTree[TreeNamespace]  
    extras: PyTree[TreeNamespace] 


class FigParamNamespace(TreeNamespace):
    """Namespace PyTree whose attributes are all `None` unless assigned.
    
    This is useful because different subclasses of `AbstractAnalysis` may call different
    plotting functions, each of which may take arbitrary keyword arguments. Thus we can 
    define defaults for any subset of these arguments in the implementation of 
    `fig_params: ClassVar[FigParamNamespace]` for the subclass, while still passing `None` to the plotting 
    functions for those parameters which do not need to be explicitly specified. 
    Likewise, the user can pass arbitrary kwargs to the `with_fig_params` method without 
    their having to be hardcoded into the subclass implementation.
    """

    # Only called if the attribute is not found in the instance `__dict__`
    def __getattr__(self, item: str) -> Any:
        if item.startswith('__'):
            # Avert issues with methods like `copy.deepcopy` which check for presence of dunder methods
            return super().__getattr__(item)
        return None
    

# Alias for constructing `FigParamNamespace` defaults in Equinox Module fields
DefaultFigParamNamespace = lambda **kwargs: eqx.field(default_factory=lambda: FigParamNamespace(**kwargs))


class _PrepOp(NamedTuple):
    name: str
    label: str
    dep_name: Optional[Union[str, Sequence[str]]]  # Dependencies to transform
    transform_func: Callable[..., Any]
    params: dict[str, Any] = {}  


class _FigOp(NamedTuple):
    name: str
    label: str
    dep_name: Optional[Union[str, Sequence[str]]] 
    is_leaf: Callable[[Any], bool]
    slice_fn: Callable[[Any, Any], Any]
    items_fn: Callable[[Any], Any]
    agg_fn: Callable[[list[PyTree], Iterable], PyTree]
    fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], dict[str, Any]]]
    params: dict[str, Any] = {}


class _FinalOp(NamedTuple):
    name: str
    label: str
    transform_func: Callable[[PyTree[go.Figure]], PyTree[go.Figure]]
    params: dict[str, Any] = {}
    is_leaf: Callable[[Any], bool] = None
    
    
"""Sentinel indicating a required dependency."""
#! TODO: Make Generic, if that makes sense -- so we can indicate the type to the user 
class _RequiredType: ...
Required = _RequiredType()
    

PARAM_SEQ_LEN_TRUNCATE = 9


def _process_param(param: Any) -> Any:
    """
    Process parameter values for serialization in the database.
    """
    # Process value based on its type
    if isinstance(param, Callable):
        return get_name_of_callable(param)
    elif isinstance(param, Mapping):
        # Preserve structure but ensure keys are strings
        return {str(mk): _process_param(mv) for mk, mv in param.items()}
    elif isinstance(param, (list, tuple)) or eqx.is_array(param):
        # Convert to list
        param_list = list(str(p) for p in param)
        if len(param_list) > PARAM_SEQ_LEN_TRUNCATE:
            return f"[{', '.join(param_list[:PARAM_SEQ_LEN_TRUNCATE])}, ..., {param_list[-1]}]"
        else:
            return f"[{', '.join(param_list)}]"
    else:
        # Simple types
        return param


def _combine_figures(
    figs_list: list[PyTree[go.Figure]], 
    items_iterated: Iterable,
) -> PyTree[go.Figure]:
    """Merge traces from multiple figures into a single one."""
    def combine_figs(*figs):
        if not figs:
            return None
        
        layout = figs[0].layout
        
        if layout.legend.traceorder == "reversed":
            layout.legend.traceorder = "grouped+reversed"
            
        if layout.legend.grouptitlefont.style is None:
            layout.legend.grouptitlefont.style = "italic"
                        
        traces = [trace for fig in figs for trace in fig.data]
        
        fig = go.Figure(data=traces, layout=layout)
        return fig 
    
    return jt.map(
        combine_figs,
        *figs_list,
        is_leaf=is_type(go.Figure),
    ) 


def _format_level_str(label: str):
    return label.replace(STRINGS.hps_level_label_sep, '-').replace('_', '')


def _format_dict_of_params(d: dict, join_str: str = ', '):
    # For constructing parts of `AbstractAnalysis.__str__`
    return join_str.join([
        f"{k}={_process_param(v)}" for k, v in d.items()
    ])


_NAME_NORMALIZATION = {"states": "data.states"}


def _normalize_name(name: str) -> str:
    return _NAME_NORMALIZATION.get(name, name)


def _axis_items_fn(axis: int, leaf: Array) -> Iterable:
        if not isinstance(leaf, Array) or axis >= leaf.ndim:
            raise ValueError(f"Combine target for axis {axis} is not Array or axis out of bounds.")
        return range(leaf.shape[axis])


def _axis_slice_fn(axis: int, node: Array, idx: int) -> Array:
        return node[(slice(None),) * axis + (idx,)]


def _level_slice_fn(node: LDict, item: Any) -> Any:
        return node[item]


def _level_items_fn(level: str, leaf: LDict) -> Iterable:
    if not LDict.is_of(level)(leaf):
            raise TypeError(f"Map target for level '{level}' is not an LDict with that label.")
    return leaf.keys()


def _reconstruct_ldict_aggregator(level: str, figs_list: list[PyTree], items_iterated: Iterable) -> LDict:
    # items_iterated here will be the keys from the LDict level
    # Rebuild the LDict using the original level label
    return LDict.of(level)(dict(zip(items_iterated, figs_list)))


def get_validation_trial_specs(task: AbstractTask):
    # TODO: Support any number of extra axes (i.e. for analyses that vmap over multiple axes in their task/model objects)
    if len(task.workspace.shape) == 3:
        #! I don't understand why/if this is necessary
        return eqx.filter_vmap(lambda task: task.validation_trials)(task)
    else:
        return task.validation_trials
    

def _extract_vmapped_kwargs_to_args(func, vmapped_dep_names: Sequence[str]):
    """Convert specified kwargs to positional args for vmapping."""
    def modified_func(data, *vmapped_deps, **remaining_kwargs):
        # Reconstruct the full kwargs dict
        full_kwargs = remaining_kwargs | dict(zip(vmapped_dep_names, vmapped_deps))
        return func(data, **full_kwargs)
    
    return modified_func


def _build_in_axes_sequence(
    dependency_axes: Mapping[str, tuple[int, ...]],
    vmapped_dep_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Build the in_axes_sequence for vmapping over inputs to `AbstractAnalysis.compute`."""
    if not dependency_axes:
        return ()
    
    max_levels = max(len(axes) for axes in dependency_axes.values())
    
    in_axes_sequence = []
    for level in range(max_levels):
        # Data axis
        data_axis = None
        if "data.states" in dependency_axes:
            state_axes = dependency_axes["data.states"]
            if level < len(state_axes):
                data_axis = AnalysisInputData(None, None, state_axes[level], None, None)
            else:
                data_axis = AnalysisInputData(None, None, None, None, None)
        else:
            data_axis = AnalysisInputData(None, None, None, None, None)
        
        # Vmapped dependency axes
        vmapped_dep_axes = []
        for dep_name in vmapped_dep_names:
            if dep_name in dependency_axes:
                dep_axes = dependency_axes[dep_name]
                if level < len(dep_axes):
                    vmapped_dep_axes.append(dep_axes[level])
                else:
                    vmapped_dep_axes.append(None)
            else:
                vmapped_dep_axes.append(None)
        
        in_axes_sequence.append((data_axis, *vmapped_dep_axes))
    
    return tuple(in_axes_sequence)


class _AnalysisVmapSpec(Module):
    """Handles specifications for vmapping over inputs to `AbstractAnalysis.compute`."""
    dependency_axes: Mapping[str, tuple[int, ...]] = eqx.field(default_factory=dict)
    vmapped_dep_names: tuple[str, ...] = ()
    in_axes_sequence: tuple[int, ...] = ()
    
    def with_dependencies(self, dep_names: Sequence[str], axes: tuple[int, ...]) -> Self:
        """Return a new VmapSpecs with additional dependencies using the same axes."""
        new_dependency_axes = dict(self.dependency_axes)
        
        # Add all dependencies with the same axes
        for dep_name in dep_names:
            new_dependency_axes[dep_name] = axes
        
        new_vmapped_dep_names = tuple(
            name for name in new_dependency_axes.keys() if name != "data.states"
        )
        
        new_in_axes_sequence = _build_in_axes_sequence(new_dependency_axes, new_vmapped_dep_names)
        
        return _AnalysisVmapSpec(
            dependency_axes=new_dependency_axes,
            vmapped_dep_names=new_vmapped_dep_names,
            in_axes_sequence=new_in_axes_sequence,
        )
        

_FinalOpKeyType = Literal['results', 'figs']


class AbstractAnalysis(Module, strict=False):
    """Component in an analysis pipeline.
    
    In `run_analysis`, multiple sets of evaluations may be performed
    prior to analysis. In particular, we may evaluate a full/large set
    of task conditions for statistical purposes, and evaluate a smaller 
    version for certain visualizations. Thus `AbstractAnalysis` 
    subclasses expect arguments `models`, `tasks`, `states`, and `hps` all 
    of which are PyTrees. The top-level structure of these PyTrees is always 
    a 
    
    Now, while it may be the case that an analysis would depend on both the 
    larger and smaller variants (in our example), we still must specify only a 
    single `variant`, since this determines the hyperparameters that are passed 
    to `analysis.save`. Thus it is assumed that all figures that result from a 
    call to some `AbstractAnalysis.make_figs` will be best associated with only
    one (and always the same one) of the eval variants.
    
    TODO: If we return the hps on a fig-by-fig basis from within `make_figs`, then 
    we could avoid this limitation.    
    
    Abstract class attributes:
        inputs: Specifies the named input ports and their default dependency types
            for this subclass of `AbstractAnalysis`.
        variant: Label of the evaluation variant this analysis uses (primarily).
    
    Abstract fields:
        conditions: In `run_analysis`, certain condition checks are performed. The 
            analysis is only run if all of the checks whose keys are in `conditions`
            are successful. For example, certain figures may only make sense to generate
            when there is system noise (i.e. multiple evals per condition), and in 
            that case we could give the condition `"any_system_noise"` to those analyses.
    """
    _exclude_fields = (
        'dependencies', 
        'conditions', 
        'fig_params', 
        'custom_dependencies', #! Should this be here?
        '_prep_ops', 
        # '_state_vmap_axes',
        '_fig_op',
        '_final_ops_by_type',
    )

    default_dependencies: AbstractClassVar[MappingProxyType[str, type[Self]]]
    conditions: AbstractVar[tuple[str, ...]]
    variant: AbstractVar[Optional[str]] 
    fig_params: AbstractVar[FigParamNamespace]
    
    # By using `strict=False`, we can define non-abstract fields, i.e. without needing to 
    # implement them trivially in subclasses. This violates the abstract-final design
    # pattern. This is intentional. If it leads to problems, I will learn from that.
    #! This means no non-default arguments in subclasses
    custom_dependencies: Mapping[str, str | Self] = eqx.field(default_factory=dict)
    _prep_ops: tuple[_PrepOp, ...] = ()
    _vmap_spec: Optional[_AnalysisVmapSpec] = None
    _fig_op: Optional[_FigOp] = None
    _final_ops_by_type: Mapping[_FinalOpKeyType, tuple[_FinalOp, ...]] = eqx.field(
        default_factory=lambda: MappingProxyType({'results': (), 'figs': ()})
    )
        
    def _compute_with_ops(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> dict[str, PyTree[Any]]:
        """Perform computations with prep-ops, vmap, and result final-ops applied."""
        
        # Transform dependencies prior to performing the analysis
        # e.g. see `after_stacking` for an example of defining a pre-op

        prepped_kwargs = self._run_prep_ops(data, kwargs)
        
        prepped_data = eqx.tree_at(
            lambda d: d.states, data, prepped_kwargs["data.states"]
        )
        del prepped_kwargs["data.states"]  
        
        if self._vmap_spec is not None:
            vmapped_deps = [prepped_kwargs.pop(name) for name in self._vmap_spec.vmapped_dep_names]
            
            compute_func = _extract_vmapped_kwargs_to_args(self.compute, self._vmap_spec.vmapped_dep_names)
            compute_func = partial(compute_func, **prepped_kwargs)
            
            result = vmap_multi(compute_func, in_axes_sequence=self._vmap_spec.in_axes_sequence)(
                prepped_data, *vmapped_deps,
            )
        else:
            compute = partial(self.compute, **prepped_kwargs)
            result = compute(prepped_data)
            
        for final_op in self._final_ops_by_type.get('results', ()):
            try:
                result = final_op.transform_func(result)
            except Exception as e:
                logger.error(f"Error during execution of final op '{final_op.name}'", exc_info=True)
                raise e
        
        return result

    def _make_figs_with_ops(
        self, 
        data: AnalysisInputData,
        result: dict[str, PyTree[Any]],
        **kwargs,
    ) -> PyTree[go.Figure]:
        """Generate figures with fig-ops and figure final-ops applied."""
        
        # Transform dependencies prior to making figures

        prepped_kwargs = self._run_prep_ops(data, kwargs)

        prepped_data = eqx.tree_at(
            lambda d: d.states, data, prepped_kwargs["data.states"]
        )
        del prepped_kwargs["data.states"]

        figs: PyTree[go.Figure] = None
        if self._fig_op is None:
            figs = self.make_figs(prepped_data, result=result, **prepped_kwargs)
        else:
            fig_op = self._fig_op

            # `make_figs` inputs will need to be sliced for different figures.
            # Prepare by amalgamating them.
            prepped_kwargs_with_results = prepped_kwargs.copy()
            prepped_kwargs_with_results["data.states"] = prepped_data.states
            prepped_kwargs_with_results["result"] = result

            # Determine which dependencies to process
            target_dep_names = self._get_target_dependency_names(
                fig_op.dep_name, prepped_kwargs_with_results, "Fig op"
            )

            dependencies_to_process: Dict[str, Any] = {}
            if target_dep_names:
                 dependencies_to_process = {name: prepped_kwargs_with_results[name] for name in target_dep_names}

            if dependencies_to_process:
                # Find the first leaf to determine items
                first_dep = next(iter(dependencies_to_process.values()))

                if first_dep is not None: 
                    try:
                        first_leaf = jt.leaves(first_dep, is_leaf=fig_op.is_leaf)[0]
                        items_to_iterate = list(fig_op.items_fn(first_leaf))

                        figs_list = []
                        for i, item in enumerate(items_to_iterate):
                            # Slice the `make_figs` inputs
                            sliced_kwargs = prepped_kwargs_with_results.copy() # Start from processed state for each slice
                            for k, v in dependencies_to_process.items():
                                sliced_kwargs[k] = jt.map(
                                    lambda x: fig_op.slice_fn(x, item) if fig_op.is_leaf(x) else x,
                                    v,
                                    is_leaf=fig_op.is_leaf
                                )

                            # Modify the `fig_params` of the `AbstractAnalysis` instance
                            analysis_for_item = self
                            if fig_op.fig_params_fn is not None:
                                modified_fig_params = fig_op.fig_params_fn(self.fig_params, i, item)
                                analysis_for_item = eqx.tree_at(
                                    lambda a: a.fig_params,
                                    self,
                                    self.fig_params | modified_fig_params
                                )

                            # Pop the `data.states` back out of the amalgamated inputs, if necessary 
                            data_for_item = data
                            if "data.states" in sliced_kwargs:
                                data_for_item = eqx.tree_at(
                                    lambda d: d.states, data, sliced_kwargs["data.states"]
                                )
                                del sliced_kwargs["data.states"]

                            # Generate a figure by passing sliced inputs to `make_figs` of the modified instance
                            slice_figs = analysis_for_item.make_figs(data_for_item, **sliced_kwargs)
                            figs_list.append(slice_figs)

                        # Potentially combine the figures into a single figure
                        if figs_list:
                             figs = fig_op.agg_fn(figs_list, items_to_iterate)
                        else:
                             logger.error(f"No figures generated by fig op for {self.name}")
                    except StopIteration:
                         logger.error(f"Could not find leaf matching predicate for fig op in dependency '{list(dependencies_to_process.keys())[0]}'. Skipping fig op.")
                    except Exception as e:
                         logger.error(f"Error during fig op execution", exc_info=True)
                         raise e
            else:
                logger.warning("Fig ops require dependencies to vary over figures, but no valid dependencies were specified.")
                
            if figs is None and self._fig_op:
                 logger.warning(f"Fig operation for {self.name} could not proceed or produced no figures.")

        for final_fig_op in self._final_ops_by_type.get('figs', ()):
            try: 
                figs = final_fig_op.transform_func(figs)
            except Exception as e:
                logger.error(f"Error during execution of final op '{final_fig_op.name}'", exc_info=True)
                raise e

        return figs

    def __call__(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> tuple[PyTree[Any], PyTree[go.Figure]]:
        """Perform analysis: compute results and generate figures."""
        result = self._compute_with_ops(data, **kwargs)
        figs = self._make_figs_with_ops(data, result, **kwargs)
        return result, figs

    @property
    def dependencies(self) -> Dict[str, type[Self] | Self | str]:
        """Get the complete mapping of dependency names to their sources.
        
        Combines default inputs with custom overrides.
        """
        return dict(self.default_dependencies) | self.custom_dependencies

    def _get_target_dependency_names(
        self,
        dep_name_spec: Optional[Union[str, Sequence[str]]],
        available_kwargs: Dict[str, Any],
        op_context: str, # e.g., "Prep-op", "Fig op"
    ) -> list[str]:
        """
        Determines the list of valid dependency names based on the specification
        and available kwargs.
        """
        target_names: list[str] = []

        if dep_name_spec is None:
            # Use all dependencies relevant to this analysis instance found in kwargs
            target_names = [k for k in self.dependencies if k in available_kwargs]
            # Process the states and `compute` results by default
            target_names.append('data.states')
            if 'result' in available_kwargs:
                target_names.append('result')
            if not target_names and self.dependencies: # Log only if dependencies were expected
                 logger.warning(f"{op_context} needs dependencies (dep_name_spec=None), but none found in kwargs.")
        else:
            # dep_name_spec = jt.map(_normalize_name, dep_name_spec)
            if isinstance(dep_name_spec, str):
                # Single dependency name
                if dep_name_spec in available_kwargs:
                    target_names = [dep_name_spec]
                else:
                    logger.warning(f"{op_context} dependency '{dep_name_spec}' not found in available kwargs.")
            elif isinstance(dep_name_spec, Sequence):
                # Sequence of dependency names
                target_names = [name for name in dep_name_spec if name in available_kwargs]
                if len(target_names) < len(dep_name_spec):
                    missing = set(dep_name_spec) - set(target_names)
                    logger.warning(f"{op_context} dependencies missing from kwargs: {missing}. Proceeding with available: {target_names}")
                if not target_names:
                    logger.warning(f"{op_context} specified dependencies {dep_name_spec}, but none were found in kwargs.")
            else:
                logger.error(f"Invalid type for {op_context} dep_name_spec: {type(dep_name_spec)}")

        return target_names

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Return kwargs to be used when instantiating dependencies.
        
        Subclasses can override this method to provide parameters for their dependencies.
        Returns a dictionary mapping dependency name to a dictionary of kwargs.
        """
        return {}

    def compute(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> dict[str, PyTree[Any]]:
        """Perform computations for the analysis. 
        
        The return value is passed as `result` to `make_figs`, and is also made available to other
        subclasses of `AbstractAnalysis` as defined in their respective`dependencies` attribute. 

        Note that the outer (task variant) `dict` level should be retained in the returned PyTree, since generally 
        a subclass that implements `compute` is implicitly available as a dependency for other subclasses
        which may depend on data for any variant.  
        """
        return 
    
    def make_figs(
        self, 
        data: AnalysisInputData,
        *,
        result: Optional[Any],
        **kwargs,
    ) -> PyTree[go.Figure]:
        """Generate figures for this analysis.
        
        Figures are returned, but are not made available to other subclasses of `AbstractAnalysis`
        which depend on the subclass implementing this method. 
        """
        return 
    
    def save_figs(
        self, 
        db_session: Session, 
        eval_info: EvaluationRecord, 
        result, 
        figs: PyTree[go.Figure],   
        hps: dict[str, PyTree[TreeNamespace]],   # dict level: variant
        model_info=None,
        dump_path: Optional[Path] = None,
        dump_formats: Sequence[str] = ("html",),
        **dependencies,
    ) -> None:
        """
        Save to disk and record in the database each figure in a PyTree of figures, for this analysis.
        """
        # `sep="_"`` switches the label dunders for single underscores, so 
        # in `_params_to_save` we can use an argument e.g. `train_pert_std` rather than `train__pert__std`
        param_keys = tree_level_labels(figs, is_leaf=is_type(go.Figure), sep="_")
        
        if dump_path is not None:
            dump_path = Path(dump_path)
            dump_path.mkdir(exist_ok=True, parents=True)
        
        figs_with_paths_flat = figs_flatten_with_paths(figs)
        
        # Construct this for reference to hps that should only vary with the task variant.
        hps_0 = jt.leaves(hps.get(self.variant, hps), is_leaf=is_type(TreeNamespace))[0]

        ops_params_dict, ops_filename_str = self._extract_ops_info()
        
        for i, (path, fig) in enumerate(figs_with_paths_flat):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))
            
            # Include fields from this instance, but only if they are JSON serializable
            field_params = {k: v for k, v in self._field_params.items() if is_json_serializable(v)}
            
            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **field_params,  # From the fields of the analysis subclass instance
                **self._params_to_save(  # Implemented by the subclass
                    hps, 
                    result=result, 
                    **path_params, 
                    **dependencies,  # Specified by the subclass `dependency_kwargs`, via `run_analysis`
                ),  
                eval_n=hps_0.eval_n,  #? Some things should always be included
            )

            if ops_params_dict:
                params['ops'] = ops_params_dict
            
            fig_record = add_evaluation_figure(
                db_session, 
                eval_info, 
                fig, 
                camel_to_snake(self.name), 
                model_records=model_info, 
                **params,
            )
            
            # Additionally dump to specified path if provided
            if dump_path is not None:                                
                # Create a unique filename using class name and hash
                filename = f"{self.name}_{self.md5_str}_{i}"

                savefig(fig, filename, dump_path, dump_formats, metadata=params)
                
                # Save parameters as YAML
                params_path = dump_path / f"{filename}.yaml"
                try:    
                    with open(params_path, 'w') as f:
                        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
                except Exception as e:
                    logger.error(f"Error saving fig dump parameters to {params_path}: {e}", exc_info=True)

    @property
    def _all_ops(self) -> tuple: 
        if self._fig_op is not None:
            fig_ops = (self._fig_op,)
        else:
            fig_ops = ()

        all_final_ops = sum(self._final_ops_by_type.values(), ())

        return self._prep_ops + fig_ops + all_final_ops

    def _extract_ops_info(self):
        """
        Extract information about all operations (prep ops and fig op).
        
        Returns:
            - ops_params_dict: Dictionary with all operations info
            - ops_filename_str: String representation for filename
        """
        ops_params_dict = {
            op.name: {k: _process_param(v) for k, v in op.params.items()}
            for op in self._all_ops
        }

        ops_filename_str = '__'.join(op.label for op in self._all_ops)

        return ops_params_dict, ops_filename_str
    
    def __str__(self) -> str:
        field_params = dict(variant=self.variant) | dict(self._non_default_field_params)
        op_params_strs = [
            f"{op.name}({_format_dict_of_params(op.params)})"
            for op in self._all_ops
        ]
        return '.'.join([
            f"{self.name}({_format_dict_of_params(field_params)})", 
            *op_params_strs
        ])
        
    def with_fig_params(self, **kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            self.fig_params | kwargs,
        )

    def after_indexing(
        self, 
        axis: int, 
        idxs: ArrayLike, 
        axis_label: Optional[str] = None,
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that slices its inputs along an axis before proceeding.
        """

        if axis_label is None:
            label = f"axis{axis}-idx{idxs}"
        else: 
            label = f"{axis_label}-idx{idxs}"

        def index_func(dep_data, **kwargs):
            return jtree.take(dep_data, idxs, axis)

        return self._add_prep_op(
            name="after_indexing",
            label=label,
            dep_name=dependency_name,
            transform_func=index_func,
            params=dict(axis=axis, idxs=idxs),
        )

    def after_map(
        self,
        func: Callable[[Any], Any], 
        is_leaf: Optional[Callable[[Any], bool]] = None, 
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that maps a function over the input PyTrees.
        """
        return self._add_prep_op(
            name="map",
            label=f"map-{get_name_of_callable(func)}",
            dep_name=dependency_name,
            transform_func=lambda dep_data, **kwargs: jt.map(func, dep_data, is_leaf=is_leaf),
            params=dict(func=func),
        )
    
    def after_transform(
        self, 
        func: Callable[..., Any],  # Must take two arguments: a PyTree, and **kwargs
        level: Optional[str | Sequence[str]] = None,
        dependency_name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that transforms its inputs with a function before
        proceeding.
        
        Applies a function to one or more `LDict` levels, or to the entire input PyTree. 
        This is less general than `after_map`.

        Args:
            func: The function to transform the inputs with.
            level: The `LDict` level to apply the transformation to. If None, the transformation is applied to the entire input PyTree.
            dependency_name: The name of the dependency to transform.
            label: The label for the transformation. If None, the label is generated from the function name and level.
        """
        obj = self 

        if level is None:
            obj = obj._add_prep_op(
                name="after_transform",
                label=f"pre-transform_{get_name_of_callable(func)}",
                dep_name=dependency_name,
                transform_func=func,
                params=dict(func=func),
            )

        else:
            if isinstance(level, str):
                levels = [level]
            else:
                levels = level

            for level in levels:
                def _transform_level(dep_data, level=level, **kwargs):  
                    return jt.map(
                        lambda node: _call_user_func(func, node, kwargs), 
                        dep_data, 
                        is_leaf=LDict.is_of(level),
                    )
                
                level_str = _format_level_str(level)

                if label is None:
                    label = f"pre-transform-{level_str}_{get_name_of_callable(func)}"

                obj = obj._add_prep_op(
                    name="after_transform",
                    label=label,
                    dep_name=dependency_name,
                    transform_func=_transform_level,
                    params=dict(level=level, transform_func=func),
                )

        return obj

    def after_unstacking(
        self, 
        axis: int, 
        level_label: str, 
        keys: Optional[Sequence[Hashable]] = None, 
        dependency_name: Optional[str] = None,
        above_level: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that unpacks an array axis into an `LDict` level.
        
        Args:
            axis: The array axis to unpack
            level_label: The label for the new LDict level
            keys: The keys to use for the LDict entries. If given, must match the length of the axis.
                By default, uses integer keys starting from zero. 
            dependency_name: Optional name of specific dependency to transform
        """
        def unpack_axis(data, **kwargs):
            def transform_array(arr):
                nonlocal keys
                if keys is None:
                    keys = range(arr.shape[axis])
                else: 
                    # Check if keys length matches the axis length
                    if arr.shape[axis] != len(keys):
                        raise ValueError(f"Length of keys ({len(keys)}) must match the length of axis {axis} ({arr.shape[axis]})")
                    
                # Move the specified axis to position 0
                arr_moved = jnp.moveaxis(arr, axis, 0)
                
                # Create an LDict with the specified label
                return LDict.of(level_label)({
                    key: slice_data 
                    for key, slice_data in zip(keys, arr_moved)
                })
            
            unstacked = jt.map(
                transform_array,
                data,
                is_leaf=eqx.is_array,
            )
            
            if above_level is not None:
                unstacked = jt.map(
                    lambda subtree: move_ldict_level_above(level_label, above_level, subtree),
                    unstacked,
                    is_leaf=is_type(LDict),
                )
                
            return unstacked
        
        return self._add_prep_op(
            name="after_unstacking",
            label=f"unstack-axis{axis}-to-{_format_level_str(level_label)}",
            dep_name=dependency_name,
            transform_func=unpack_axis,
            params=dict(axis=axis, level_label=level_label, above_level=above_level), #, keys=keys),
        )

    def after_stacking(self, level: str, dependency_name: Optional[str] = None) -> Self:
        """
        Returns a copy of this analysis that stacks its inputs along an `LDict` PyTree level before proceeding.

        This is useful when we have a PyTree of results with an `LDict` level representing 
        the values across some variable we want to visually compare, and our analysis 
        uses a plotting function that compares across the first axis of input arrays.
        By stacking first, we collapse the `LDict` level into the first axis so that the 
        plotting function will compare (e.g. colour differently) across the variable.
        
        Args:
            level: The label of the `LDict` level in the PyTree to stack by. 
            dependency_name: Optional name of the specific dependency to stack.
                If None, will stack all dependencies listed in self.dependencies.
            
        Returns:
            A copy of this analysis with stacking operation and updated parameters
        """
        # Define the stacking function
        def stack_dependency(dep_data, **kwargs):
            return jt.map(
                lambda d: jtree.stack(list(d.values())),
                dep_data,
                is_leaf=LDict.is_of(level),
            )
        
        modified_analysis = eqx.tree_at(
            lambda obj: (obj.colorscale_key, obj.colorscale_axis, obj.fig_params),
            self,
            (
                level,
                0,
                self.fig_params | dict(legend_title=get_label_str(level)),
            ),
            is_leaf=is_none,
        )
        
        return modified_analysis._add_prep_op(
            name="after_stacking",
            label=f"stack_{_format_level_str(level)}",
            dep_name=dependency_name,
            transform_func=stack_dependency,
            params=dict(level=level),
        )
    
    def after_level_to_top(
        self, 
        label: str, 
        is_leaf: Callable[[Any], bool] = LDict.is_of('var'),
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that will transpose `LDict` levels of its inputs.

        This is useful when our analysis uses a plotting function that compares across 
        the outer PyTree level, but for whatever reason this level is not already 
        the outer level of our results PyTree.
        """
        def transpose_dependency(dep_data, **kwargs):
            return {
                variant_label: ldict_level_to_top(label, dep_data[variant_label], is_leaf=is_leaf)
                for variant_label in dep_data
            }
        
        return self._add_prep_op(
            name="after_level_to_top",
            label=f"{_format_level_str(label)}_to-top",
            dep_name=dependency_name,
            transform_func=transpose_dependency,
            params=dict(label=label),
        )

    def after_subdict_at_level(
        self, 
        level: str, 
        keys: Optional[Sequence[Hashable]] = None, 
        idxs: Optional[Sequence[int]] = None,
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that keeps certain keys of an `LDict` level before proceeding.

        Either `keys` or `idxs` must be provided, but not both: `keys` specifies the exact keys to keep,
        whereas `idxs` specifies the indices of the keys to keep in terms of their ordering in the `LDict`.
        """

        if keys is not None and idxs is not None:
            raise ValueError("Cannot provide both `keys` and `idxs`.")
        
        if keys is None and idxs is None:
            raise ValueError("Must provide either `keys` or `idxs`.")

        label = f"subdict-at-{_format_level_str(level)}_"
        if keys is not None:
            select_func = lambda d: subdict(d, keys)
            label += ','.join(keys)
        elif idxs is not None:
            select_func = lambda d: subdict(d, [list(d.keys())[i] for i in idxs])
            label += f"idxs-{','.join(str(i) for i in idxs)}"

        return self.after_transform(
            func=select_func, 
            level=level, 
            dependency_name=dependency_name,
            label=label,
        )
        
    def vmap(self, axes: int | Sequence[int], dependency_names: str | Sequence[str]) -> Self:
        """
        Returns a copy of this analysis that vectorizes its `compute` method over one or more axes
        of one of its inputs.
        
        Arguments:
            axes: The axes of `data.states` to vectorize over, expressed in natural form (e.g. 
                to map over the first three batch axes, pass `axes=(0, 1, 2)`).
        """
        if isinstance(dependency_names, str):
            dependency_names = (dependency_names,)
        
        if isinstance(axes, int):
            axes = (axes,)
        else:
            axes = tuple(axes)
        
        # normalized_dep_names = [_normalize_name(name) for name in dependency_names]
        normalized_dep_names = dependency_names
        
        sequential_axes = natural_to_sequential_axes(axes)

        if not all(isinstance(axis, int) for axis in sequential_axes):
            raise ValueError("`axes` must be an integer or a sequence of integers.")
        
        if self._vmap_spec is None:
            vmap_spec = _AnalysisVmapSpec()
        else:
            vmap_spec = self._vmap_spec
        
        return eqx.tree_at(
            lambda obj: obj._vmap_spec,
            self,
            vmap_spec.with_dependencies(normalized_dep_names, sequential_axes),
            is_leaf=is_none,
        )
        
    # @property
    # def _data_vmap_multi_axes(self):
    #     if self._state_vmap_axes is None:
    #         # We shouldn't arrive here
    #         return None 
    #     else:
    #         return tuple(
    #             (AnalysisInputData(None, None, i, None, None),)
    #             for i in self._state_vmap_axes
    #         )
        
    def and_transform_results(
        self,
        func: Callable[..., Any],
        level: Optional[str] = None,
        label: Optional[str] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Self:
        """Returns a copy of this analysis that transforms its PyTree of results.

        The transformation occurs prior to the generation of figures, thus affects them.
        """
        return self._then_transform(
            op_type='results',
            func=func,
            level=level,
            label=label,
            is_leaf=is_leaf,
        )
    
    def map_figs_at_level(
        self, 
        level: str, 
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], FigParamNamespace]] = None,
    ) -> Self: 
        """
        Returns a copy of this analysis that maps over the input PyTrees, down to a certain `LDict` level.

        This is useful when e.g. the analysis calls a plotting function that expects a two-level PyTree, 
        but we've evaluated a deeper PyTree of states, where the two levels are inner. 
        """
        return self._change_fig_op(
            name="map_figs_at_level",
            label=f"map_figs_at-{_format_level_str(level)}",
            dep_name=dependency_name,
            is_leaf=LDict.is_of(level),
            slice_fn=_level_slice_fn,
            items_fn=partial(_level_items_fn, level),
            # Use the new aggregator specific to mapping
            agg_fn=partial(_reconstruct_ldict_aggregator, level),
            fig_params_fn=fig_params_fn, 
            params=dict(level=level),
        )

    def map_figs_by_axis(
        self, 
        axis: int, 
        output_level_label: str, 
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], FigParamNamespace]] = None,
    ) -> Self:
        """Returns a copy of this analysis that maps over a given axis of the input PyTree(s).
        
        This is useful when we want to produce a separate figure for each element along an array axis. 

        Args:
            axis: The axis to map over
            output_level_label: The label of the `LDict` level to create for the output figures
        """
        # TODO: combined `map_at_level` with `after_unstacking` so the user can control the 
        # keys of the resulting output LDict level
        return self._change_fig_op(
            name="map_figs_by_axis",
            label=f"map_figs_by-axis{axis}",
            dep_name=dependency_name,
            is_leaf=eqx.is_array,
            slice_fn=partial(_axis_slice_fn, axis),
            items_fn=partial(_axis_items_fn, axis),
            fig_params_fn=fig_params_fn,
            agg_fn=partial(_reconstruct_ldict_aggregator, output_level_label),
            params=dict(axis=axis),
        )
    
    def combine_figs_by_axis(
        self, 
        axis: int, 
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], FigParamNamespace]] = None
    ) -> Self:
        """
        Returns a copy of this analysis that will merge individual figures generated by slicing along
        the specified axis of the arrays in the dependency PyTree(s).

        This is useful when we want to include an additional dimension of comparison in a figure. 
        For example, our plotting function may already compare across the first axis of the input 
        arrays, or the outer level of the input PyTree; but perhaps we also want a secondary 
        comparison across a different axis of the input arrays.
        
        Args:
            axis: The axis to slice and combine along
            dependency_name: Optional name of specific dependency to slice.
                If None, will slice all dependencies listed in self.dependencies.
        """
        return self._change_fig_op(
            name="combine_figs_by_axis",
            label=f"combine_by-axis{axis}",
            dep_name=dependency_name,
            is_leaf=eqx.is_array,
            slice_fn=partial(_axis_slice_fn, axis),
            items_fn=partial(_axis_items_fn, axis),
            fig_params_fn=fig_params_fn,
            # Use the default aggregator that matches the new signature
            agg_fn=_combine_figures,
            params=dict(axis=axis),
        )
        
    def combine_figs_by_level(
        self, 
        level: str, 
        dependency_name: Optional[str] = None,
        fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], FigParamNamespace]] = None
    ) -> Self:
        """
        Returns a copy of this analysis that will merge individual figures generated by iterating over
        the keys of an LDict level in the dependency PyTree(s).

        This is useful when we want to include an additional dimension of comparison in a figure. 
        For example, our plotting function may already compare across the first axis of the input 
        arrays, or the outer level of the input PyTree; but perhaps we also want a secondary 
        comparison across a different level of the input PyTree.
        
        Args:
            level: The LDict level to iterate over and combine across
            dependency_name: Optional name of specific dependency to iterate over.
                If None, will iterate over all dependencies listed in self.dependencies.
        """
        return self._change_fig_op(
            name="combine_figs_by_level",
            label=f"combine_by-{_format_level_str(level)}",
            dep_name=dependency_name,
            is_leaf=LDict.is_of(level),
            slice_fn=_level_slice_fn,
            items_fn=partial(_level_items_fn, level),
            fig_params_fn=fig_params_fn,
             # Use the default aggregator that matches the new signature
            agg_fn=_combine_figures,
            params=dict(level=level),
        )

    def then_transform_figs(
        self,
        func: Callable[..., Any],
        level: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that transforms its output PyTree of figures
        """
        return self._then_transform(
            op_type='figs',
            func=func,
            level=level,
            label=label,
            is_leaf=is_type(go.Figure),
        )

    def _then_transform(
        self, 
        op_type: _FinalOpKeyType,
        func: Callable[..., Any], 
        level: Optional[str] = None,
        label: Optional[str] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Self:

        if label is None:
            label = f"post-transform-{op_type}_{get_name_of_callable(func)}"

        if level is not None:
            # Apply the transformation leafwise across the `level` LDict level;
            # e.g. suppose there are two keys in the `level` LDict, then the transformation
            # will be applied to 2-tuples of figures; following the transformation, reconsistute
            # the `level` LDict with the transformed figures.
            @wraps(func)
            def _transform_func(tree):
                _Tuple = jtree.make_named_tuple_subclass("ColumnTuple")
                
                def _transform_level(ldict_node):
                    if not LDict.is_of(level)(ldict_node):
                        return ldict_node
                    
                    zipped = jtree.zip_(
                        *ldict_node.values(), 
                        is_leaf=is_leaf, 
                        zip_cls=_Tuple,
                    )
                    transformed = jt.map(func, zipped, is_leaf=is_type(_Tuple))
                    unzipped = jtree.unzip(transformed, tuple_cls=_Tuple)
                    return LDict.of(level)(dict(zip(ldict_node.keys(), unzipped)))

                return jt.map(
                    _transform_level,
                    tree, 
                    is_leaf=LDict.is_of(level)
                )

            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                label=label,
                transform_func=_transform_func,
                params=dict(level=level, transform_func=func),
                is_leaf=is_leaf,
            )
        else:
            return self._add_final_op(
                op_type=op_type,
                name=f"then_transform_{op_type}",
                label=label,
                transform_func=func,
                params=dict(transform_func=func),
                is_leaf=is_leaf,
            )

    def _add_prep_op(
            self, 
            name: str,
            label: str,
            dep_name: Optional[str], 
            transform_func: Callable,
            params: Optional[Dict[str, Any]] = None,
    ) -> Self:
        # If the transform consumes extra dependencies, ensure the analysis
        # instance knows about them so the graph builder evaluates them.
        port_map = getattr(transform_func, "_cwd_port_map", {})

        if port_map:
            new_custom_deps = dict(self.custom_dependencies)
            for global_lbl, port_key in port_map.items():
                if port_key not in self.dependencies:
                    new_custom_deps[port_key] = global_lbl  # reference original label

            analysis_with_deps = eqx.tree_at(
                lambda a: a.custom_dependencies,
                self,
                new_custom_deps,
            )
        else:
            analysis_with_deps = self

        return eqx.tree_at(
            lambda a: a._prep_ops,
            analysis_with_deps,
            analysis_with_deps._prep_ops + (_PrepOp(
                name=name,
                label=label,
                dep_name=dep_name, 
                transform_func=transform_func,
                params=params,
            ),)
        )
    
    def _add_final_op(
        self,
        op_type: _FinalOpKeyType,
        name: str,
        label: str,
        transform_func: Callable,
        params: Optional[Dict[str, Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Self:
        current_ops = self._final_ops_by_type.get(op_type, ())
        new_op = _FinalOp(
            name=name,
            label=label,
            transform_func=transform_func,
            params=params,
            is_leaf=is_leaf,
        )
        updated_ops_for_type = current_ops + (new_op,)
        
        # Create a new dictionary for _final_ops_by_type to ensure immutability if needed by equinox
        new_final_ops_by_type = self._final_ops_by_type.copy()
        new_final_ops_by_type[op_type] = updated_ops_for_type
        
        return eqx.tree_at(
            lambda a: a._final_ops_by_type,
            self,
            new_final_ops_by_type,
        )

    def _change_fig_op(self, **kwargs) -> Self:
        return eqx.tree_at(
            lambda a: a._fig_op,
            self,
            _FigOp(**kwargs),
            is_leaf=is_none,
        )
                      
    @cached_property
    def _field_params(self):
        # TODO: Inherit from dependencies? 
        return get_dataclass_fields(
            self, 
            exclude=AbstractAnalysis._exclude_fields,
            include_internal=False,
        )

    @cached_property
    def _non_default_field_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of fields that have non-default values.
        Works without knowing field names in advance.
        """
        result = {}
        
        # Get all dataclass fields for this instance
        for field in dataclasses.fields(self):
            # Exclude `variant` since we explicitly include it first, in dump file names
            if field.name in AbstractAnalysis._exclude_fields or field.name == "variant":
                continue
            
            # Skip fields that are marked as subclass-internal
            if field.metadata.get('internal', False):
                continue

            current_value = getattr(self, field.name)
            
            # Check if this field has a default value defined
            has_default = field.default is not dataclasses.MISSING
            has_default_factory = field.default_factory is not dataclasses.MISSING
            
            if has_default and current_value != field.default:
                # Field has a different value than its default
                result[field.name] = current_value
            elif has_default_factory:
                # For default_factory fields, we can't easily tell if the value
                # was explicitly provided, so we include the current value
                # This is an approximation - we'll include fields with default_factory
                result[field.name] = current_value
            elif not has_default and not has_default_factory:
                # Field has no default, so it must have been provided
                result[field.name] = current_value
                
        return result

    @cached_property
    def md5_str(self):
        """An md5 hash string that identifies this analysis.
        
        The hash is computed from the analysis parameter values and not the instance itself.
        """
        ops_params, _ = self._extract_ops_info()
        params = {**ops_params, **self._field_params}
        return get_md5_hexdigest(params)
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        """Additional parameters to save.
        
        Note that `**kwargs` here may not only contain the dependencies, but that `save` 
        passes the key-value pairs of parameters inferred from the `figs` PyTree. 
        """
        return dict()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _run_prep_ops(self, data: AnalysisInputData, kwargs: Dict[str, Any]):
        """Apply all prep-ops in sequence, consuming extra CallWithDeps deps."""

        prepped_kwargs = kwargs.copy() # Start with original kwargs for modification
        prepped_kwargs['data.states'] = data.states
        
        for prep_op in self._prep_ops:
            dep_names_to_process = self._get_target_dependency_names(
                prep_op.dep_name, prepped_kwargs, "Prep-op"
            )

            for name in dep_names_to_process:
                try:
                    prepped_kwargs[name] = _call_user_func(
                        prep_op.transform_func,
                        prepped_kwargs[name],
                        kwargs,
                    )
                except Exception as e:
                    logger.error(
                        f"Error applying prep_op transform to '{name}'", exc_info=True
                    )
                    raise e

            # Pop any CallWithDeps-only dependencies that are not part of the analysis interface
            port_map = getattr(prep_op.transform_func, "_cwd_port_map", {})
            for port_key in port_map.values():
                if port_key not in self.dependencies:
                    prepped_kwargs.pop(port_key, None)

        return prepped_kwargs


AnalysisDependenciesType: TypeAlias = MappingProxyType[str, type[AbstractAnalysis]]


class _DummyAnalysis(AbstractAnalysis):
    """An empty analysis, for debugging."""
    default_dependencies: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = None
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    def compute(self, data: AnalysisInputData, **kwargs) -> PyTree[Any]:
        print(tree_level_labels(next(iter(data.states.values()))))
        return None
    
    def make_figs(self, data: AnalysisInputData, **kwargs) -> PyTree[go.Figure]:
        return None


# --------------------------------------------------------------------------- #
# Helper: safely forward kwargs to user-supplied callbacks                   #
# --------------------------------------------------------------------------- #

def _call_user_func(func, dep_data, extra_kwargs):
    """Invoke *func* with *dep_data* and the subset of *extra_kwargs* it accepts.

    If *func* originated from CallWithDeps we may need to translate
    user-visible dependency labels (e.g. "hidden_states_pca") that are
    present in *extra_kwargs* into the unique, private port names
    (e.g. "__cwd_1") that the wrapper will look up internally.  The
    mapping is stored on the wrapper object as ``_cwd_port_map``.
    """
    port_map: dict[str, str] | None = getattr(func, "_cwd_port_map", None)
    port_names = set(port_map.values()) if port_map else set()

    sig = inspect.signature(func)

    # Fast path: the function accepts **kwargs; forward everything.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(dep_data, **extra_kwargs)

    # Otherwise filter down to only the kwargs explicitly listed.
    filtered = {
        k: v for k, v in extra_kwargs.items() 
        if k in sig.parameters or k in port_names
    }
    return func(dep_data, **filtered)


class CallWithDeps:
    """Wrap *func* so that selected dependency values are routed into its
    positional and keyword parameters when used inside `after_transform` or
    similar prep-ops.

    Usage
    -----
    ```python
    .after_transform(
        CallWithDeps("pca")(lambda pca, x: pca.batch_transform(x)),
        dependency_name="data.states",
    )
    ```

    Positional mapping: each entry in *pos_deps* is either a dependency name
    (looked up in the `**kwargs` dict passed by the pipeline) or the literal
    value ``None`` which indicates where the *current* dependency (i.e.
    `dependency_name` being transformed) should be inserted.  If ``None`` is
    absent, the current dependency is appended after the mapped ones.

    Keyword mapping: ``CallWithDeps(z="other_dep")`` will forward the value of
    "other_dep" to the keyword parameter ``z`` of *func*.
    """

    _counter = 0

    def __init__(self, *pos_deps: str | None, **kw_deps: str):
        self.pos_deps = pos_deps
        self.kw_deps = kw_deps

    def __call__(self, func: Callable):
        # Build a stable mapping from global dependency label -> unique internal port name
        label_set = [d for d in self.pos_deps if d is not None] + list(self.kw_deps.values())
        label_to_port: dict[str, str] = {}
        for lbl in label_set:
            if lbl not in label_to_port:
                CallWithDeps._counter += 1
                label_to_port[lbl] = f"__cwd_{CallWithDeps._counter:x}"

        @wraps(func)
        def wrapper(dep_data, **all_kwargs):
            # Build positional arguments
            pos_args = []
            for dep in self.pos_deps:
                if dep is None:
                    pos_args.append(dep_data)
                else:
                    port = label_to_port[dep]
                    try:
                        pos_args.append(all_kwargs[port])
                    except KeyError:
                        raise KeyError(f"Dependency '{dep}' not found in kwargs for CallWithDeps wrapper") from None

            if None not in self.pos_deps:
                pos_args.append(dep_data)

            # Build keyword arguments mapped explicitly
            mapped_kwargs = {}
            for param, dep_name in self.kw_deps.items():
                port = label_to_port[dep_name]
                try:
                    mapped_kwargs[param] = all_kwargs[port]
                except KeyError:
                    raise KeyError(f"Dependency '{dep_name}' not found for keyword param '{param}'") from None

            # Expose map on wrapper BEFORE returning
            return func(*pos_args, **mapped_kwargs)
        wrapper._cwd_port_map = label_to_port

        return wrapper


