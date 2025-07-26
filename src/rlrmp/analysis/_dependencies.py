"""
Interpret analysis graphs, to avoid re-computing shared dependencies.

This is useful because it allows us to specify the dependencies as class attributes of the 
subclasses of `AbstractAnalysis`, and they will automatically be computed only once for all 
of the requested analyses (in `run_analysis.py`).

An alternative is to allow for statefulness of the analysis classes. Then analyses callback to 
their dependencies, and also memoize their results so that repeat work is avoided. However, 
I've decided to use `eqx.Module` and stick with a stateless solution. So we need to explicitly
parse the graph.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
import hashlib
import inspect
import json
from typing import Optional, Set, Dict, Any, ClassVar, Callable

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

from rlrmp.analysis.analysis import (
    AbstractAnalysis, AnalysisInputData, _format_dict_of_params, RequiredInput, OptionalInput,
    _DataField, ConstantInput, FigParamNamespace, DefaultFigParamNamespace, AnalysisDefaultInputsType
)
from types import MappingProxyType
from rlrmp.misc import get_md5_hexdigest
from rlrmp.tree_utils import prefix_expand


logger = logging.getLogger(__name__)


class _DataForwarder(AbstractAnalysis):
    """Forwards a single attribute of `AnalysisInputData`.

    This node exists only to integrate `Data.<attr>` references into the
    dependency graph.  It performs no computation other than returning the
    chosen attribute and produces no figures.
    
    NOTE: See also `_DataField` which is used to specify the attribute to forward.
    """

    # Name of the attribute (e.g. "states", "models", …) to be forwarded
    attr: str = ""
    where: Optional[Callable] = None
    is_leaf: Optional[Callable[[Any], bool]] = None

    # No dependencies of its own
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType({})  # type: ignore
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = None
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    # Pure forwarding
    def compute(self, data: AnalysisInputData, **kwargs):  # noqa: D401
        value = getattr(data, self.attr)
        if self.where is not None:
            value = jt.map(self.where, value, is_leaf=self.is_leaf)
        return value

    def make_figs(self, data: AnalysisInputData, *, result=None, **kwargs):  # noqa: D401
        # nothing plotted
        return None

    def __post_init__(self):  # noqa: D401
        if not self.attr:
            raise ValueError("_DataForwarder.attr must be provided")


def param_hash(params: Dict[str, Any]) -> str:
    """Create a hash of parameter values to uniquely identify dependency configurations."""
    # Convert params to a stable string representation and hash it
    params_formatted = _format_dict_of_params(params)
    param_str = json.dumps(params_formatted, sort_keys=True)
    return get_md5_hexdigest(param_str)


def get_params_for_dep_class(analysis, dep_class):
    """Get parameters for a dependency based on its class."""
    # Check if the class exists in dep_params
    dep_params = getattr(analysis, 'dependency_params', {})
    return dep_params.get(dep_class, {})


def resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup=None):
    """Resolve a dependency source to an analysis instance and create a graph node ID.
    
    Args:
        analysis: The analysis instance requesting the dependency
        dep_name: The name of the dependency port  
        dep_source: Either a class type, string reference, analysis instance, or ConstantInput
        dependency_lookup: Optional dict for resolving string references
    Returns:
        tuple: (node_id, params, analysis_instance) or None for ConstantInput
    """
    # Handle ConstantInput - skip dependency resolution
    if isinstance(dep_source, ConstantInput):
        return None
        
    # Handle required-but-missing dependencies early
    if dep_source is RequiredInput:
        raise ValueError(
            f"Dependency '{dep_name}' for analysis '{analysis.name}' is marked as RequiredInput but was not provided. "
            "Pass it via `custom_inputs` on that analysis instance, or reference an entry in the module-level "
            "`DEPENDENCIES` dict and point to it by name from `custom_inputs`."
        )
    
    # Handle optional inputs that weren't provided
    if dep_source is OptionalInput:
        raise ValueError(
            f"Dependency '{dep_name}' for analysis '{analysis.name}' is marked as OptionalInput and should not "
            "appear in the computation graph unless explicitly overridden in `custom_inputs`."
        )
    
    # Handle forwarding of attributes from AnalysisInputData via the `Data` sentinel
    if isinstance(dep_source, _DataField):
        # Treat each attribute as a unique forwarding analysis node, including any transform
        analysis_instance = _DataForwarder(
            attr=dep_source.attr,
            where=dep_source.where,
            is_leaf=dep_source.is_leaf,
        )
        node_id = analysis_instance.md5_str
        return node_id, {}, analysis_instance
    
    class_params = analysis.dependency_kwargs().get(dep_name, {})
    # Recursively resolve string dependencies
    if dep_source is None:
        raise ValueError(f"Dependency '{dep_name}' is None")
    if isinstance(dep_source, str):
        if dependency_lookup is not None and dep_source in dependency_lookup:
            dep_instance = dependency_lookup[dep_source]
            node_id = dep_source  # Use the string key as node_id for deduplication
            return node_id, class_params, dep_instance
        else:
            raise ValueError(f"String dependency '{dep_source}' could not be resolved. Provide dependency_lookup with all available keys.")
    if isinstance(dep_source, type):
        # Class type - create instance and use its hash
        field_params = get_params_for_dep_class(analysis, dep_source)
        params = {**field_params, **class_params}
        analysis_instance = dep_source(**params)
        node_id = analysis_instance.md5_str
        return node_id, params, analysis_instance
    else:
        # Already an analysis instance - use its hash directly
        if dependency_lookup is not None:
            for k, v in dependency_lookup.items():
                if v is dep_source:
                    node_id = k
                    return node_id, class_params, dep_source
        node_id = dep_source.md5_str
        return node_id, class_params, dep_source


def _process_dependency_sources(analysis, dep_name, dep_sources, dependency_lookup, callback):
    """Helper to iterate over dependency sources and apply callback to non-ConstantInput deps."""
    for dep_source in dep_sources:
        dep_result = resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup)
        if dep_result is not None:  # Skip ConstantInput
            callback(dep_result)


def build_dependency_graph(analyses: Sequence[AbstractAnalysis], dependency_lookup=None) -> tuple[dict[str, set], dict]:
    """Build a directed acyclic graph of analysis dependencies."""
    graph = defaultdict(set)  # Maps node_id -> set of dependency node_ids
    nodes = {}  # Maps md5 hash (str) -> (analysis_instance, params)

    def _validate_signature(func, required_names: list[str], func_label: str, analysis_name: str):
        """Check that *func* lists all *required_names* unless it has **kwargs."""
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return  # **kwargs catches all
        missing = [n for n in required_names if n not in sig.parameters]
        if missing:
            raise ValueError(
                f"Analysis '{analysis_name}' expects parameters {missing} in `{func_label}` but they are not present. "
                "Add them to the signature or accept **kwargs."
            )

    def _add_deps(analysis: AbstractAnalysis):
        """Recursively add analysis and its dependencies to the graph."""
        md5_id = analysis.md5_str
        if md5_id not in nodes:
            nodes[md5_id] = (analysis, {})

            # Validate that compute/make_figs accept required parameters
            deps_list = list(analysis.inputs.keys())
            _validate_signature(analysis.compute, deps_list, "compute", analysis.name)
            _validate_signature(analysis.make_figs, deps_list + ["result"], "make_figs", analysis.name)
            
            # Recursively process all flattened dependencies
            for dep_name, dep_sources in analysis._flattened_inputs.items():
                def add_dependency(dep_result):
                    dep_node_id, params, dep_instance = dep_result
                    if dep_instance.md5_str not in nodes:
                        nodes[dep_instance.md5_str] = (dep_instance, params)
                        _add_deps(dep_instance)
                
                _process_dependency_sources(
                    analysis, dep_name, dep_sources, dependency_lookup, add_dependency
                )

    # Add all analyses from dependency lookup and main analysis list
    for analysis in (dependency_lookup or {}).values():
        _add_deps(analysis)
    for analysis in analyses:
        _add_deps(analysis)
    
    # Build graph edges from analyses to their direct dependencies
    for analysis in analyses:
        for dep_name, dep_sources in analysis._flattened_inputs.items():
            def add_edge(dep_result):
                _, _, dep_instance = dep_result
                analysis_node_id = analysis.md5_str
                dep_node_id = dep_instance.md5_str
                graph[analysis_node_id].add(dep_node_id)
                # Ensure every edge target is also present in `nodes`
                if dep_node_id not in nodes:
                    nodes[dep_node_id] = (dep_instance, {})
            
            _process_dependency_sources(
                analysis, dep_name, dep_sources, dependency_lookup, add_edge
            )

    # Build graph edges for transitive dependencies (dependencies of dependencies)
    for node_id, (dep_instance, _) in list(nodes.items()):
        for subdep_name, subdep_sources in dep_instance._flattened_inputs.items():
            def add_transitive_edge(dep_result):
                _, _, sub_dep_instance = dep_result
                sub_dep_id = sub_dep_instance.md5_str
                graph[node_id].add(sub_dep_id)
                if sub_dep_id not in nodes:
                    nodes[sub_dep_id] = (sub_dep_instance, {})

            _process_dependency_sources(
                dep_instance, subdep_name, subdep_sources, dependency_lookup, add_transitive_edge
            )

    # Ensure all nodes have graph entries (even if no dependencies)
    for node_id in nodes:
        if node_id not in graph:
            graph[node_id] = set()
    return dict(graph), nodes


def topological_sort(graph: dict[str, Set[str]]) -> list[str]:
    """Return dependencies in order they should be computed."""
    visited = set()
    temp_marks = set()
    order = []
    
    def visit(node: str):
        if node in temp_marks:
            raise ValueError(f"Circular dependency detected at node {node}")
        if node in visited:
            return
            
        temp_marks.add(node)
        
        # Visit all dependencies first
        for dep in graph.get(node, set()):
            visit(dep)
            
        temp_marks.remove(node)
        visited.add(node)
        order.append(node)
    
    # Visit all nodes
    for node in graph:
        if node not in visited:
            visit(node)
            
    return order


def _resolve_expand_to_dependencies(
    analysis: AbstractAnalysis, 
    dep_kwargs: dict[str, Any], 
    dependency_lookup: dict[str, AbstractAnalysis]
) -> dict[str, Any]:
    """Apply prefix expansion to inputs that were wrapped in ExpandTo.
    
    All dependencies (including ExpandTo sources and targets) are already resolved.
    This just applies the prefix_expand transformation.
    """
    expand_specs = analysis._expand_to_specs
    if not expand_specs:
        return dep_kwargs
    
    for input_name, expand_objects in expand_specs.items():
        expand_to = expand_objects[0]  # Assume one ExpandTo per input for now
        
        # Get target structure (already resolved as dependency)
        if isinstance(expand_to.target, str):
            target_structure = dep_kwargs[expand_to.target]
        elif isinstance(expand_to.target, _DataField):
            target_key = f"_{input_name}_expand_target"
            target_structure = dep_kwargs[target_key]
        else:
            raise TypeError(f"Invalid ExpandTo target type: {type(expand_to.target)}")
        
        # Apply prefix expansion to already-resolved source dependencies
        dep_kwargs[input_name] = prefix_expand(
            dep_kwargs[input_name],  # Already resolved ExpandTo.source dependencies
            target_structure,
            is_leaf=expand_to.is_leaf,
            is_leaf_prefix=expand_to.is_leaf_prefix
        )
    
    return dep_kwargs


def _reconstruct_dependencies(
    analysis: AbstractAnalysis, 
    computed_results: dict[str, Any], 
    dependency_lookup: dict[str, AbstractAnalysis]
) -> dict[str, Any]:
    """Reconstruct PyTree dependencies for an analysis from computed leaf results."""
    dep_kwargs = {}
    
    # Collect and reconstruct each dependency PyTree
    for dep_name, dep_sources in analysis._flattened_inputs.items():
        leaf_results = []
        
        # Gather results for all leaves in this dependency
        for dep_source in dep_sources:
            if isinstance(dep_source, ConstantInput):
                # Add ConstantInput values directly to leaf_results
                leaf_results.append(dep_source.value)
            else:
                dep_result = resolve_dependency_node(
                    analysis, dep_name, dep_source, dependency_lookup=dependency_lookup
                )
                if dep_result is not None:
                    _, _, dep_instance = dep_result
                    dep_hash = dep_instance.md5_str
                    if dep_hash in computed_results:
                        leaf_results.append(computed_results[dep_hash])
                    else:
                        raise RuntimeError(
                            f"Missing dependency result for '{dep_name}' in analysis '{analysis.__class__.__name__}'. "
                            f"Dependency hash '{dep_hash}' not found in computed results."
                        )
        
        # Ensure all dependencies were resolved before reconstruction
        if len(leaf_results) != len(dep_sources):
            missing = len(dep_sources) - len(leaf_results)
            raise RuntimeError(
                f"Failed to resolve {missing}/{len(dep_sources)} dependencies for "
                f"'{dep_name}' in analysis '{analysis.__class__.__name__}'. "
                f"This indicates a bug in dependency resolution or circular dependencies."
            )
        
        # Reconstruct PyTree from leaf results
        tree_def = analysis.input_treedefs[dep_name]
        dep_kwargs[dep_name] = jt.unflatten(tree_def, leaf_results)
    
    # Resolve ExpandTo dependencies after regular dependencies are reconstructed
    dep_kwargs = _resolve_expand_to_dependencies(analysis, dep_kwargs, dependency_lookup)
    
    return dep_kwargs


def compute_dependency_results(
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    custom_dependencies: Optional[Dict[str, AbstractAnalysis]] = None,
    **kwargs,
) -> list[dict[str, PyTree[Any]]]:
    """Compute all dependencies in correct order.
    
    Args:
        analyses: Analysis instances to process (sequence or dict)
        data: Input data for analysis  
        custom_dependencies: Optional dict of custom dependency instances (from DEPENDENCIES)
        **kwargs: Additional baseline dependencies
    """
    if custom_dependencies is None:
        custom_dependencies = {}
    analyses_list = list(analyses.values())
    dependency_lookup = custom_dependencies | analyses
    
    # Build computation graph and determine execution order
    graph, dep_instances = build_dependency_graph(analyses_list, dependency_lookup=dependency_lookup)
    comp_order = topological_sort(graph)
    
    # Create reverse lookup for better logging
    hash_to_key = {}
    for key, instance in dependency_lookup.items():
        hash_to_key[instance.md5_str] = key
        
    # Track which nodes are leaf analyses to avoid redundant dependency reconstruction
    leaf_node_ids = {analysis.md5_str for analysis in analyses_list}
    leaf_dependencies = {}  # Cache reconstructed dependencies for leaf analyses
        
    baseline_kwargs = kwargs.copy()
    computed_results = {}
    
    # Execute all dependencies in topological order
    for node_id in comp_order:
        dep_instance, params = dep_instances[node_id]
        
        # Reconstruct dependencies and add instance parameters
        dep_kwargs = baseline_kwargs.copy()
        dep_kwargs.update(params)
        
        reconstructed_deps = _reconstruct_dependencies(dep_instance, computed_results, dependency_lookup)
        dep_kwargs.update(reconstructed_deps)
        
        if node_id in leaf_node_ids:
            # Cache dependency reconstruction for leaf analyses to avoid redundant work
            leaf_dependencies[node_id] = reconstructed_deps
        
        # Log computation for non-trivial analyses
        if node_id in hash_to_key:
            log_name = hash_to_key[node_id]
        else:
            log_name = f"{dep_instance.__class__.__name__} ({dep_instance.md5_str})"
        
        if not log_name.startswith("_DataForwarder"):
            logger.info(f"Computing analysis node: {log_name}")
        
        # Execute analysis and store result
        result = dep_instance._compute_with_ops(data, **dep_kwargs)
        computed_results[node_id] = result
    
    # Assemble final results for each requested analysis
    all_dependency_results = []
    for analysis in analyses_list:
        dependency_results = baseline_kwargs.copy()
        
        # Add the analysis result itself
        analysis_hash = analysis.md5_str
        if analysis_hash in computed_results:
            dependency_results['result'] = computed_results[analysis_hash]
        
        # Reuse cached dependency reconstruction for leaf analyses
        dependency_results.update(leaf_dependencies[analysis_hash])
        all_dependency_results.append(dependency_results)
    
    return all_dependency_results