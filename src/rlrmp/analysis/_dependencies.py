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

from rlrmp.analysis.analysis import (
    AbstractAnalysis, AnalysisInputData, _format_dict_of_params, Required,
    _DataField, FigParamNamespace, DefaultFigParamNamespace, AnalysisDependenciesType
)
from types import MappingProxyType
from rlrmp.misc import get_md5_hexdigest


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helper analysis to forward attributes from AnalysisInputData                #
# --------------------------------------------------------------------------- #


class _DataForwarder(AbstractAnalysis):
    """Forwards a single attribute of `AnalysisInputData`.

    This node exists only to integrate `Data.<attr>` references into the
    dependency graph.  It performs no computation other than returning the
    chosen attribute and produces no figures.
    """

    # Name of the attribute (e.g. "states", "models", …) to be forwarded
    attr: str = ""
    where: Optional[Callable] = None
    is_leaf: Optional[Callable[[Any], bool]] = None

    # No dependencies of its own
    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType({})  # type: ignore
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
        dep_source: Either a class type, string reference, or analysis instance
        dependency_lookup: Optional dict for resolving string references
    Returns:
        tuple: (node_id, params, analysis_instance)
    """
    # Handle required-but-missing dependencies early
    if dep_source is Required:
        raise ValueError(
            f"Dependency '{dep_name}' for analysis '{analysis.name}' is marked as Required but was not provided. "
            "Pass it via `custom_inputs` on that analysis instance, or reference an entry in the module-level "
            "`DEPENDENCIES` dict and point to it by name from `custom_inputs`."
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


def build_dependency_graph(analyses: Sequence[AbstractAnalysis], dependency_lookup=None) -> tuple[dict[str, set], dict]:
    graph = defaultdict(set)
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

    # Only store each analysis instance by its md5 hash. All lookups and graph edges use md5 hashes only.
    def _add_deps(analysis: AbstractAnalysis):
        md5_id = analysis.md5_str
        if md5_id not in nodes:
            nodes[md5_id] = (analysis, {})

            deps_list = list(analysis.inputs.keys())
            _validate_signature(
                analysis.compute,
                deps_list,
                "compute",
                analysis.name,
            )
            _validate_signature(
                analysis.make_figs,
                deps_list + ["result"],
                "make_figs",
                analysis.name,
            )
            for dep_name, dep_source in analysis.inputs.items():
                dep_node_id, params, dep_instance = resolve_dependency_node(
                    analysis, dep_name, dep_source, dependency_lookup=dependency_lookup
                )
                if dep_instance.md5_str not in nodes:
                    nodes[dep_instance.md5_str] = (dep_instance, params)
                    _add_deps(dep_instance)

    for analysis in (dependency_lookup or {}).values():
        _add_deps(analysis)
    for analysis in analyses:
        _add_deps(analysis)
    for analysis in analyses:
        for dep_name, dep_source in analysis.inputs.items():
            _, _, dep_instance = resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup=dependency_lookup)
            analysis_node_id = analysis.md5_str
            dep_node_id = dep_instance.md5_str
            graph[analysis_node_id].add(dep_node_id)
            # Ensure every edge target is also present in `nodes`
            if dep_node_id not in nodes:
                nodes[dep_node_id] = (dep_instance, {})

    for node_id, (dep_instance, _) in list(nodes.items()):
        for subdep_name, subdep_source in dep_instance.inputs.items():
            _, _, sub_dep_instance = resolve_dependency_node(
                dep_instance,
                subdep_name,
                subdep_source,
                dependency_lookup=dependency_lookup,
            )
            sub_dep_id = sub_dep_instance.md5_str
            graph[node_id].add(sub_dep_id)
            if sub_dep_id not in nodes:
                nodes[sub_dep_id] = (sub_dep_instance, {})

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


def compute_dependency_results(
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    custom_dependencies: Optional[Dict[str, AbstractAnalysis]] = None,
    **kwargs,
) -> dict:
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
    
    graph, dep_instances = build_dependency_graph(analyses_list, dependency_lookup=dependency_lookup)
    comp_order = topological_sort(graph)
    
    # Create a reverse lookup from md5 hash to key for better logging
    hash_to_key = {}
    for key, instance in dependency_lookup.items():
        hash_to_key[instance.md5_str] = key
        
    baseline_kwargs = kwargs.copy()
    computed_results = {}
    
    for node_id in comp_order:
        dep_instance, params = dep_instances[node_id]
        # Only pass baseline kwargs and the relevant dependencies under their local names
        dep_kwargs = baseline_kwargs.copy()
        for dep_name, dep_source in dep_instance.inputs.items():
            _, _, sub_dep_instance = resolve_dependency_node(
                dep_instance, 
                dep_name, 
                dep_source, 
                dependency_lookup=dependency_lookup,
            )
            sub_dep_hash = sub_dep_instance.md5_str
            if sub_dep_hash in computed_results:
                dep_kwargs[dep_name] = computed_results[sub_dep_hash]
        dep_kwargs.update(params)
        
        # Use key if available, otherwise use class name and hash
        if node_id in hash_to_key:
            log_name = hash_to_key[node_id]
        else:
            log_name = f"{dep_instance.__class__.__name__} ({dep_instance.md5_str})"
        
        if not log_name.startswith("_DataForwarder"):
            logger.info(f"Computing analysis node: {log_name}")
        
        result = dep_instance._compute_with_ops(data, **dep_kwargs)
        computed_results[node_id] = result
    
    all_dependency_results = []
    for analysis in analyses_list:
        dependency_results = baseline_kwargs.copy()
        # Add the result of this analysis itself
        analysis_hash = analysis.md5_str
        if analysis_hash in computed_results:
            dependency_results['result'] = computed_results[analysis_hash]
        # Add dependencies
        for dep_name, dep_source in analysis.inputs.items():
            _, _, dep_instance = resolve_dependency_node(analysis, dep_name, dep_source, dependency_lookup=dependency_lookup)
            dep_hash = dep_instance.md5_str
            if dep_hash in computed_results:
                dependency_results[dep_name] = computed_results[dep_hash]
        all_dependency_results.append(dependency_results)
    
    return all_dependency_results