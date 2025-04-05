from collections.abc import Callable, Hashable, Sequence
import dataclasses
from functools import cached_property
import inspect
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Optional, Dict, Self, Union
from pathlib import Path
import yaml

import equinox as eqx
from equinox import AbstractVar, AbstractClassVar, Module
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import ArrayLike, PyTree, Array
import plotly.graph_objects as go
from sqlalchemy.orm import Session

from jax_cookbook import is_type, is_none
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.database import EvaluationRecord, add_evaluation_figure, savefig
from rnns_learn_robust_motor_policies.tree_utils import subdict, tree_level_labels
from rnns_learn_robust_motor_policies.misc import camel_to_snake, get_dataclass_fields, is_json_serializable
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
    dep_name: Optional[Union[str, Sequence[str]]] # Allow str or Sequence[str]
    transform_func: Callable[[Any], Any]


class _FigOp(NamedTuple):
    dep_name: Optional[Union[str, Sequence[str]]] # Allow str or Sequence[str]
    is_leaf: Callable[[Any], bool]
    slice_fn: Callable[[Any, Any], Any]
    items_fn: Callable[[Any], Any]
    agg_fn: Callable[[list[PyTree], Iterable], PyTree]
    fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], dict[str, Any]]]


def _combine_figures(
    figs_list: list[PyTree[go.Figure]], 
    items_iterated: Iterable,
) -> PyTree[go.Figure]:
    def combine_figs(*figs):
        if not figs:
            return None
        
        # Use first figure as base (make a copy to avoid modifying original)
        base_fig = go.Figure(figs[0])
        
        # Add traces from other figures
        for fig in figs[1:]:
            if fig is not None:
                base_fig.add_traces(fig.data)
        
        return base_fig
    
    return jt.map(
        combine_figs,
        *figs_list,
        is_leaf=is_type(go.Figure),
    ) 


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
        dependencies: Specifies the subclasses of `AbstractAnalysis`
            whose results are needed for this subclass of `AbstractAnalysis`.
        variant: Label of the evaluation variant this analysis uses (primarily).
    
    Abstract fields:
        conditions: In `run_analysis`, certain condition checks are performed. The 
            analysis is only run if all of the checks whose keys are in `conditions`
            are successful. For example, certain figures may only make sense to generate
            when there is system noise (i.e. multiple evals per condition), and in 
            that case we could give the condition `"any_system_noise"` to those analyses.
    """
    _exclude_fields = ('dependencies', 'conditions', 'fig_params', '_prep_ops', '_fig_op')

    dependencies: AbstractClassVar[MappingProxyType[str, type[Self]]]
    conditions: AbstractVar[tuple[str, ...]]
    variant: AbstractVar[Optional[str]] 
    fig_params: AbstractVar[FigParamNamespace]

    # By using `strict=False`, we can define some private fields without needing to 
    # implement them trivially in subclasses. This violates the abstract-final design
    # pattern. This is intentional. If it leads to problems, I will learn from that.
    #! This means no non-default arguments in subclasses
    _prep_ops: tuple[_PrepOp, ...] = ()
    _fig_op: Optional[_FigOp] = None

    def __call__(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> tuple[PyTree[Any], PyTree[go.Figure]]:

        # Transform dependencies prior to performing the analysis
        # e.g. see `after_stacking` for an example of defining a pre-op
        processed_kwargs = kwargs.copy() # Start with original kwargs for modification
        for prep_op in self._prep_ops:
            dep_names_to_process = self._get_target_dependency_names(
                prep_op.dep_name, processed_kwargs, "Prep-op"
            )

            # Apply transformation to identified dependencies
            for name in dep_names_to_process:
                 try:
                     # Modify the dictionary being processed
                     processed_kwargs[name] = prep_op.transform_func(processed_kwargs[name])
                 except Exception as e:
                     logger.error(f"Error applying prep_op transform to '{name}': {e}", exc_info=True)

        result = self.compute(data, **processed_kwargs)

        figs: PyTree[go.Figure] = None
        if self._fig_op is None:
            figs = self.make_figs(data, result=result, **processed_kwargs)
        else:
            fig_op = self._fig_op

            # Determine which dependencies to process
            target_dep_names = self._get_target_dependency_names(
                fig_op.dep_name, processed_kwargs, "Fig op"
            )

            dependencies_to_process: Dict[str, Any] = {}
            if target_dep_names:
                 dependencies_to_process = {name: processed_kwargs[name] for name in target_dep_names}

            if dependencies_to_process:
                # Find the first leaf to determine items
                first_dep = next(iter(dependencies_to_process.values()))

                if first_dep is not None: 
                    try:
                        first_leaf = jt.leaves(first_dep, is_leaf=fig_op.is_leaf)[0]
                        items_to_iterate = list(fig_op.items_fn(first_leaf))

                        figs_list = []
                        for i, item in enumerate(items_to_iterate):
                            # Create the sliced kwargs based on the potentially modified processed_kwargs
                            sliced_kwargs = processed_kwargs.copy() # Start from processed state for each slice
                            for k, v in dependencies_to_process.items():
                                sliced_kwargs[k] = jt.map(
                                    lambda x: fig_op.slice_fn(x, item) if fig_op.is_leaf(x) else x,
                                    v,
                                    is_leaf=fig_op.is_leaf
                                )

                            analysis_for_item = self
                            if fig_op.fig_params_fn is not None:
                                modified_fig_params = fig_op.fig_params_fn(self.fig_params, i, item)
                                analysis_for_item = eqx.tree_at(
                                    lambda a: a.fig_params,
                                    self,
                                    self.fig_params | modified_fig_params
                                )

                            # Pass sliced_kwargs to make_figs
                            slice_figs = analysis_for_item.make_figs(data, result=result, **sliced_kwargs)
                            figs_list.append(slice_figs)

                        if figs_list:
                             figs = fig_op.agg_fn(figs_list, items_to_iterate)
                        else:
                             logger.error(f"No figures generated by fig op for {self.name}")
                    except StopIteration:
                         logger.error(f"Could not find leaf matching predicate for fig op in dependency '{list(dependencies_to_process.keys())[0]}'. Skipping fig op.")
                    except Exception as e:
                         logger.error(f"Error during fig op execution: {e}", exc_info=True)

            if figs is None and self._fig_op:
                 logger.warning(f"Fig operation for {self.name} could not proceed or produced no figures.")

        return result, figs

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
            if not target_names and self.dependencies: # Log only if dependencies were expected
                 logger.warning(f"{op_context} needs dependencies (dep_name_spec=None), but none found in kwargs.")
        elif isinstance(dep_name_spec, str):
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
    ) -> Optional[PyTree[Any]]:
        """Perform computations for the analysis. 
        
        The return value is passed as `result` to `make_figs`, and is also made available to other
        subclasses of `AbstractAnalysis` as defined in their respective`dependencies` attribute. 
        """
        return 
    
    def make_figs(
        self, 
        data: AnalysisInputData,
        *,
        result: Optional[Any],
        **kwargs,
    ) -> Optional[PyTree[go.Figure]]:
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
        hps: PyTree[TreeNamespace], 
        model_info=None,
        dump_path: Optional[Path] = None,
        dump_formats: list[str] = ["html"],
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
        hps_0 = jt.leaves(hps[self.variant], is_leaf=is_type(TreeNamespace))[0]
        
        for i, (path, fig) in enumerate(figs_with_paths_flat):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))
            
            # Include fields from this instance, but only if they are JSON serializable
            field_params = {k: v for k, v in self._field_params.items() if is_json_serializable(v)}
            
            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **field_params,  # From the fields of this subclass
                **self._params_to_save(
                    hps, 
                    result=result, 
                    **path_params, 
                    **dependencies,  # Specified by the subclass `dependency_kwargs`, via `run_analysis`
                ),  
                eval_n=hps_0.eval_n,  #? Some things should always be included
            )
            
            add_evaluation_figure(
                db_session, 
                eval_info, 
                fig, 
                camel_to_snake(self.name), 
                model_records=model_info, 
                **params,
            )
            
            # Include any fields that have non-default values in the filename; 
            # this serves to distinguish different instances of the same analysis,
            # according to the kwargs passed by the user upon instantiation.
            # TODO: Exclude non-determining fields like `legend_title` 
            # TODO: (could group all the figure layout kwargs under a single field and exclude it)
            non_default_field_params_str = '__'.join([
                f"{k}-{v}" for k, v in self._non_default_field_params.items()
            ])
            
            # Additionally dump to specified path if provided
            if dump_path is not None:                                
                # Create a unique filename
                analysis_name = camel_to_snake(self.name)
                filename = f"{analysis_name}__{self.variant}__{non_default_field_params_str}__{i}"
                
                savefig(fig, filename, dump_path, dump_formats)
                
                # Save parameters as YAML
                params_path = dump_path / f"{filename}.yaml"
                with open(params_path, 'w') as f:
                    yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        params = dict(self._non_default_field_params)
        params = dict(variant=self.variant) | params
        non_default_field_params_str = ', '.join([
            f"{k}={v}" for k, v in params.items()
        ])
        return f"{self.name}({non_default_field_params_str})"

    def with_fig_params(self, **kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            self.fig_params | kwargs,
        )

    def after_indexing(self, axis: int, idxs: ArrayLike, dependency_name: Optional[str] = None) -> Self:
        """
        Returns a copy of this analysis that slices its inputs along an axis before proceeding.
        """

        return self._add_prep_op(
            dep_name=dependency_name,
            transform_func=lambda dep_data: jtree.take(dep_data, axis, idxs),
        )
    
    def after_transform_level(
        self, 
        level: str | Sequence[str], 
        transform_func: Optional[Callable[[LDict], Any]],
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that transforms its inputs at an `LDict` level before proceeding.
        """

        if isinstance(level, str):
            level = [level]

        obj = self
        for current_level in level:
            # Define transform_level inside the loop with the correct closure
            def _transform_level(dep_data, level=current_level):  # Capture current_level by value
                return jt.map(transform_func, dep_data, is_leaf=LDict.is_of(level))
            
            obj = obj._add_prep_op(
                dep_name=dependency_name,
                transform_func=_transform_level,
            )

        return obj
        
    def after_unstacking(
        self, 
        axis: int, 
        label: str, 
        keys: Optional[Sequence[Hashable]] = None, 
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that unpacks an array axis into an `LDict` level.
        
        Args:
            axis: The array axis to unpack
            label: The label for the new LDict level
            keys: The keys to use for the LDict entries. If given, must match the length of the axis.
                By default, uses integer keys starting from zero. 
            dependency_name: Optional name of specific dependency to transform
        """
        def unpack_axis(data):
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
                return LDict.of(label)({
                    key: slice_data 
                    for key, slice_data in zip(keys, arr_moved)
                })
            
            return jt.map(
                transform_array,
                data,
                is_leaf=eqx.is_array,
            )
        
        return self._add_prep_op(
            dep_name=dependency_name,
            transform_func=unpack_axis,
        )
    
    def after_subdict_at_level(
        self, 
        level: str, 
        keys: Sequence[Hashable], 
        dependency_name: Optional[str] = None,
    ) -> Self:
        """
        Returns a copy of this analysis that keeps certain  an `LDict` level before proceeding.
        """

        return self.after_transform_level(
            level, 
            select_func=lambda d: subdict(d, keys), 
            dependency_name=dependency_name,
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
        def stack_dependency(dep_data):
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
            dep_name=dependency_name,
            transform_func=stack_dependency
        )
    
    def after_level_to_top(self, label: str, dependency_name: Optional[str] = None) -> Self:
        """
        Returns a copy of this analysis that will transpose `LDict` levels of its inputs.

        This is useful when our analysis uses a plotting function that compares across 
        the outer PyTree level, but for whatever reason this level is not already 
        the outer level of our results PyTree.
        """
        def transpose_dependency(dep_data):
            return jt.transpose(
                jt.structure(dep_data, is_leaf=LDict.is_of(label)),
                None,
                dep_data,
            )
        
        return self._add_prep_op(
            dep_name=dependency_name,
            transform_func=transpose_dependency,
        )

    def map_at_level(self, level: str, dependency_name: Optional[str] = None) -> Self: 
        """
        Returns a copy of this analysis that maps over the input PyTrees, down to a certain `LDict` level.

        This is useful when e.g. the analysis calls a plotting function that expects a two-level PyTree, 
        but we've evaluated a deeper PyTree of states, where the two levels are inner. 
        """
        # Define items_fn and slice_fn (same as combine_figs_by_level)
        _is_leaf_level = LDict.is_of(level)
        def _level_items_fn(leaf: LDict) -> Iterable:
            if not _is_leaf_level(leaf):
                 raise TypeError(f"Map target for level '{level}' is not an LDict with that label.")
            return leaf.keys()
        def _level_slice_fn(node: LDict, item: Any) -> Any:
             return node[item]

        # Define the specific aggregator for reconstructing the LDict
        def _reconstruct_ldict_aggregator(figs_list: list[PyTree], items_iterated: Iterable) -> LDict:
            # items_iterated here will be the keys from the LDict level
            # Rebuild the LDict using the original level label
            return LDict.of(level)(dict(zip(items_iterated, figs_list)))

        return self._change_fig_op(
            dep_name=dependency_name,
            is_leaf=_is_leaf_level,
            slice_fn=_level_slice_fn,
            items_fn=_level_items_fn,
            # Use the new aggregator specific to mapping
            agg_fn=_reconstruct_ldict_aggregator,
            fig_params_fn=None, # Mapping usually doesn't need per-item param changes
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
        # Define items_fn and slice_fn locally
        def _axis_items_fn(leaf: Array) -> Iterable:
             if not isinstance(leaf, Array) or axis >= leaf.ndim:
                 raise ValueError(f"Combine target for axis {axis} is not Array or axis out of bounds.")
             return range(leaf.shape[axis])
        def _axis_slice_fn(node: Array, item: int) -> Array:
             return node[(slice(None),) * axis + (item,)]

        return self._change_fig_op(
            dep_name=dependency_name,
            is_leaf=eqx.is_array,
            slice_fn=_axis_slice_fn,
            items_fn=_axis_items_fn,
            fig_params_fn=fig_params_fn,
            # Use the default aggregator that matches the new signature
            agg_fn=_combine_figures,
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
        # Define items_fn and slice_fn locally
        _is_leaf_level = LDict.is_of(level)
        def _level_items_fn(leaf: LDict) -> Iterable:
            if not _is_leaf_level(leaf):
                 raise TypeError(f"Combine target for level '{level}' is not an LDict with that label.")
            return leaf.keys()
        def _level_slice_fn(node: LDict, item: Any) -> Any:
             return node[item]

        return self._change_fig_op(
            dep_name=dependency_name,
            is_leaf=_is_leaf_level,
            slice_fn=_level_slice_fn,
            items_fn=_level_items_fn,
            fig_params_fn=fig_params_fn,
             # Use the default aggregator that matches the new signature
            agg_fn=_combine_figures,
        )

    def _add_prep_op(self, dep_name: Optional[str], transform_func: Callable) -> Self:
        return eqx.tree_at(
            lambda a: a._prep_ops,
            self,
            self._prep_ops + (_PrepOp(dep_name=dep_name, transform_func=transform_func),)
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
        # TODO: Inherit from dependencies? e.g. if we depend on `BestReplicateStates`, maybe we should include `i_replicate` from there
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
            if field.name in AbstractAnalysis._exclude_fields:
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

    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        """Additional parameters to save.
        
        Note that `**kwargs` here may not only contain the dependencies, but that `save` 
        passes the key-value pairs of parameters inferred from the `figs` PyTree. 
        Thus for example `train_pert_std` is explicitly referred to in the argument list of 
        `plant_perts.Effector_ByEval._params_to_save`.
        """
        return dict()

    @property
    def name(self) -> str:
        return self.__class__.__name__
