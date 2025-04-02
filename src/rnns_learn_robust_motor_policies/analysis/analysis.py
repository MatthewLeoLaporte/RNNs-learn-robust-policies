from collections.abc import Callable, Sequence
import dataclasses
from functools import cached_property
import inspect
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Dict, Self
from pathlib import Path
import yaml

import equinox as eqx
from equinox import AbstractVar, AbstractClassVar, Module
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


# #! TODO: Replace with something other than `Module`, and move `colorscale_key` and `colorscale_axis`
# #! to `AbstractAnalysis`, so that the user can pass arbitrary kwargs to the plotting function
# class FigParams(Module):
#     """Parameters used for figure generation."""
#     legend_title: Optional[str] = None
#     legend_labels: Optional[Sequence[str]] = None
    
#     # Additional params that might be used by different subclasses
#     n_curves_max: Optional[int] = None
#     curves_mode: Optional[str] = None

#     def with_updates(self, **kwargs):
#         """Create a new TreeNamespace with specific fields updated."""
#         # Start with current values
#         current_values = {
#             field.name: getattr(self, field.name) 
#             for field in dataclasses.fields(self)
#         }
        
#         return TreeNamespace(**current_values | kwargs)


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
    #! TODO: Multiple user-specified dependencies
    dep_name: Optional[str]  # Name of dependency to operate on, or None for all
    transform_func: Callable[[Any], Any]  # Function to transform the dependency


class _FigOp(NamedTuple):
    dep_name: Optional[str]  # Name of dependency to operate on, or None for all
    is_leaf: Callable[[Any], bool]  
    slice_fn: Callable[[Any, Any], Any]  
    items_fn: Callable[[Any], Any]  # Function to get items to iterate over by calls to `make_figs`
    fig_params_fn: Optional[Callable[[FigParamNamespace, int, Any], dict[str, Any]]]  # Modify fig_params for each iteration


def _combine_figures(figs_list):
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
    _exclude_fields = ('dependencies', 'conditions', 'fig_params', '_prep_ops', '_fig_ops')

    conditions: AbstractVar[tuple[str, ...]]
    dependencies: AbstractClassVar[MappingProxyType[str, type[Self]]]
    variant: AbstractVar[Optional[str]] 
    fig_params: AbstractVar[FigParamNamespace]

    # By using `strict=False`, we can define some private fields without needing to 
    # implement them trivially in subclasses. This violates the abstract-final design
    # pattern. This is intentional. If it leads to problems, I will learn from that.
    #! This means no non-default arguments in subclasses
    _prep_ops: tuple[_PrepOp, ...] = ()
    _fig_ops: tuple[_FigOp, ...] = ()

    def with_fig_params(self, **kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            self.fig_params | kwargs,
        )
    
    def __call__(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> tuple[PyTree[Any], PyTree[go.Figure]]:
        
        # Transform dependencies prior to performing the analysis
        # e.g. see `after_stacking` for an example of defining a pre-op
        for prep_op in self._prep_ops:
            if prep_op.dep_name is None:
                # Transform all dependencies
                for dependency_name in self.dependencies.keys():
                    assert dependency_name in kwargs, ""
                    # Apply to the subtrees below the top (variant) level
                    kwargs[dependency_name] = {
                        variant_label: prep_op.transform_func(variant_kwarg)
                        for variant_label, variant_kwarg in kwargs[dependency_name].items()
                    }
            else:
                if prep_op.dep_name in kwargs:
                    kwargs[prep_op.dep_name] = {
                        variant_label: prep_op.transform_func(variant_kwarg)
                        for variant_label, variant_kwarg in kwargs[prep_op.dep_name].items()
                    }
                else:
                    msg = f"Prep-op cannot be performed: {prep_op.dep_name} not found in kwargs"
                    raise ValueError(msg)

        result = self.compute(data, **kwargs)

        # Handle figure operations
        if not self._fig_ops:
            # No figure operations, just call make_figs normally
            figs = self.make_figs(data, result=result, **kwargs)
        else:
            # Execute the figure operations
            for fig_op in self._fig_ops:
                # Determine which dependencies to process
                if fig_op.dep_name is None:
                    dependencies_to_process = {k: v for k, v in kwargs.items() if k in self.dependencies}
                else:
                    if fig_op.dep_name not in kwargs:
                        raise ValueError(f"Figure operation cannot be performed: {fig_op.dep_name} not found in kwargs")
                    dependencies_to_process = {fig_op.dep_name: kwargs[fig_op.dep_name]}
                
                # Get the first leaf node that matches our leaf predicate
                first_dep = next(iter(dependencies_to_process.values()))
                first_leaf = jt.leaves(first_dep, is_leaf=fig_op.is_leaf)[0]
                
                # Get the items to iterate over
                items_to_iterate = fig_op.items_fn(first_leaf)
                
                # Generate figures for eac43h item
                figs_list = []
                for i, item in enumerate(items_to_iterate):
                    # Each call to make_figs receives a single element along some level/axis/...
                    sliced_kwargs = kwargs.copy()
                    for k, v in dependencies_to_process.items():
                        sliced_kwargs[k] = jt.map(
                            lambda x: fig_op.slice_fn(x, item) if fig_op.is_leaf(x) else x,
                            v,
                            is_leaf=fig_op.is_leaf
                        )
                    
                    # Create a modified analysis with custom fig_params for this item, if provided
                    #? TODO: Could pass `fig_params` as a dependency instead of storing as a field
                    analysis_for_item = self
                    if fig_op.fig_params_fn is not None:
                        modified_fig_params = fig_op.fig_params_fn(i, item)
                        analysis_for_item = eqx.tree_at(
                            lambda a: a.fig_params,
                            self,
                            self.fig_params | modified_fig_params
                        )
                    
                    # Generate figures for this slice using the modified analysis
                    slice_figs = analysis_for_item.make_figs(data, result=result, **sliced_kwargs)
                    figs_list.append(slice_figs)
                
                # Combine figures
                figs = _combine_figures(figs_list)

        return result, figs
        
    def compute(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ) -> Optional[PyTree[Any]]:
        return 
    
    def make_figs(
        self, 
        data: AnalysisInputData,
        *,
        result: Optional[Any],
        **kwargs,
    ) -> Optional[PyTree[go.Figure]]:
        return 
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        """Additional parameters to save.
        
        Note that `**kwargs` here may not only contain the dependencies, but that `save` 
        passes the key-value pairs of parameters inferred from the `figs` PyTree. 
        Thus for example `train_pert_std` is explicitly referred to in the argument list of 
        `plant_perts.Effector_ByEval._params_to_save`.
        """
        return dict()

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Return kwargs to be used when instantiating dependencies.
        
        Subclasses can override this method to provide parameters for their dependencies.
        Returns a dictionary mapping dependency name to a dictionary of kwargs.
        """
        return {}

    def save_figs(
        self, 
        db_session: Session, 
        eval_info: EvaluationRecord, 
        result, 
        figs: PyTree[go.Figure],   
        hps: PyTree[TreeNamespace], 
        model_info=None,
        dump_path: Optional[Path] = None,
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
                
                savefig(fig, filename, dump_path, ["html"])
                
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

    def after_indexing(self, axis: int, idxs: ArrayLike, dependency_name: Optional[str] = None) -> Self:
        """
        Returns a copy of this analysis that slices its inputs along an axis before proceeding.
        """

        return self._add_prep_op(
            dep_name=dependency_name,
            transform_func=lambda dep_data: jtree.take(dep_data, axis, idxs),
        )
    
    def after_subset_at_level(self, level: str, keys: Sequence, dependency_name: Optional[str] = None) -> Self:
        """
        Returns a copy of this analysis that slices its inputs along an `LDict` PyTree level before proceeding.
        """

        def subset_level(dep_data):
            return jt.map(
                lambda d: subdict(d, keys),
                dep_data,
                is_leaf=LDict.is_of(level),
            )
        
        return self._add_prep_op(
            dep_name=dependency_name,
            transform_func=subset_level,
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
        
    def _add_prep_op(self, dep_name: Optional[str], transform_func: Callable) -> Self:
        return eqx.tree_at(
            lambda a: a._prep_ops,
            self,
            self._prep_ops + (_PrepOp(dep_name=dep_name, transform_func=transform_func),)
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
        wrapped_fig_params_fn = None
        if fig_params_fn:
            wrapped_fig_params_fn = lambda i, item: fig_params_fn(self.fig_params, i, item)

        fig_op = _FigOp(
            dep_name=dependency_name,
            is_leaf=eqx.is_array,
            slice_fn=lambda x, i: x[(slice(None),) * axis + (i,)],
            items_fn=lambda x: range(x.shape[axis]),
            fig_params_fn=wrapped_fig_params_fn,
        )

        return self._add_fig_op(fig_op)
        
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
        wrapped_fig_params_fn = None
        if fig_params_fn:
            wrapped_fig_params_fn = lambda i, item: fig_params_fn(self.fig_params, i, item)

        fig_op = _FigOp(
            dep_name=dependency_name,
            is_leaf=LDict.is_of(level),
            slice_fn=lambda x, key: x[key],
            items_fn=lambda x: x.keys(),
            fig_params_fn=wrapped_fig_params_fn,
        )
        
        return self._add_fig_op(fig_op)

    def _add_fig_op(self, fig_op: _FigOp) -> Self:
        return eqx.tree_at(
            lambda a: a._fig_ops,
            self,
            self._fig_ops + (fig_op,)
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