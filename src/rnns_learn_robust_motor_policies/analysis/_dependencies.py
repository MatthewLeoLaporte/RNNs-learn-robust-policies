"""
Interpret analysis graphs, to avoid re-computing shared dependencies.

This is useful because it allows us to specify the dependencies as class attributes of the 
subclasses of `AbstractAnalysis`, and they will automatically be computed only once for all 
of the requested analyses (in `run_analysis.py`).

An alternative is to allow for statefulness of the analysis classes. Then analyses callback to 
their dependencies, and also memoize their results so that repeat work is avoided. However, 
I've decided to use `eqx.Module` and stick with a stateless solution. So we need to explicitly
parse the graph.

Written with the help of Claude 3.5 Sonnet.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
import hashlib
import json
from typing import Optional, Set, Dict, Any

import equinox as eqx
import jax.tree as jt

from rnns_learn_robust_motor_policies.analysis.analysis import (
    AbstractAnalysis, AnalysisInputData, _format_dict_of_params
)
from rnns_learn_robust_motor_policies.misc import get_md5_hexdigest


logger = logging.getLogger(__name__)


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


def resolve_dependency_node(analysis, dep_name, dep_source):
    """Resolve a dependency source to an analysis instance and create a graph node ID.
    
    Args:
        analysis: The analysis instance requesting the dependency
        dep_name: The name of the dependency port  
        dep_source: Either a class type, string reference, or analysis instance
        
    Returns:
        tuple: (node_id, params, analysis_instance)
    """
    class_params = analysis.dependency_kwargs().get(dep_name, {})
    
    if isinstance(dep_source, str):
        # This shouldn't happen if dependencies are properly resolved beforehand
        raise ValueError(f"String dependency '{dep_source}' should be resolved before dependency graph building")
    elif isinstance(dep_source, type):
        # Class type - create instance and use its hash
        field_params = get_params_for_dep_class(analysis, dep_source)
        params = {**field_params, **class_params}
        analysis_instance = dep_source(**params)
        node_id = (dep_name, analysis_instance.md5_str)
        return node_id, params, analysis_instance
    else:
        # Already an analysis instance - use its hash directly
        node_id = (dep_name, dep_source.md5_str)
        return node_id, class_params, dep_source


def build_dependency_graph(analyses: Sequence[AbstractAnalysis]) -> tuple[dict[str, Set[str]], dict]:
    """Build dependency graph with parameter-specific nodes.
    
    Each dependency with unique parameters gets a unique node in the graph.
    Only include dependencies for analyses whose conditions are met.
    """
    graph = defaultdict(set)
    nodes = {}  # Maps node_id -> (analysis_instance, params)
    
    # TODO: Filter analyses by their conditions
    # analyses_to_process = [a for a in analyses if conditions_are_met(a)]

    def _add_deps(analysis: AbstractAnalysis):        
        for dep_name, dep_source in analysis.dependencies.items():
            node_id, params, dep_instance = resolve_dependency_node(
                analysis, dep_name, dep_source
            )
            
            if node_id not in nodes:
                # Record the mapping
                nodes[node_id] = (dep_instance, params)
                # Recursively add subdependencies
                _add_deps(dep_instance)
        
    for analysis in analyses:  
        _add_deps(analysis)

    # Build the actual graph edges
    for analysis in analyses:
        for dep_name, dep_source in analysis.dependencies.items():
            node_id, _, _ = resolve_dependency_node(analysis, dep_name, dep_source)
            # The analysis itself doesn't have a node_id, but we need to track its dependencies
            # Actually, let's add analysis nodes too for completeness
            analysis_node_id = ("analysis", analysis.md5_str)
            graph[analysis_node_id].add(node_id)
    
    # Add dependency-to-dependency edges  
    for node_id, (dep_instance, _) in nodes.items():
        for subdep_name, subdep_source in dep_instance.dependencies.items():
            subdep_node_id, _, _ = resolve_dependency_node(dep_instance, subdep_name, subdep_source) 
            graph[node_id].add(subdep_node_id)

    # Ensure all nodes exist in graph (even leaf nodes)
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
    
    # Combine lookup sources: custom_dependencies (DEPENDENCIES) + all_analyses_lookup (ALL_ANALYSES)
    dependency_lookup = custom_dependencies | analyses
    
    # Resolve any string dependencies in the analyses
    if dependency_lookup:
        resolved_analyses = []
        for analysis in analyses_list:
            resolved_deps = {}
            for dep_name, dep_source in analysis.dependencies.items():
                if isinstance(dep_source, str):
                    if dep_source in dependency_lookup:
                        resolved_deps[dep_name] = dependency_lookup[dep_source]
                    else:
                        available_keys = list(dependency_lookup.keys())
                        raise ValueError(f"Dependency with key '{dep_source}' not found. Available keys: {available_keys}")
                else:
                    resolved_deps[dep_name] = dep_source
            
            if resolved_deps != dict(analysis.dependencies):
                # Create new analysis instance with resolved dependencies
                analysis = eqx.tree_at(
                    lambda a: a.custom_dependencies,
                    analysis, 
                    resolved_deps,
                )
            resolved_analyses.append(analysis)
        analyses_list = resolved_analyses

    # Build dependency graph with resolved dependencies
    graph, dep_instances = build_dependency_graph(analyses_list)
    
    # Get computation order
    comp_order = topological_sort(graph)
    
    # Compute dependencies in order
    results = kwargs.copy()
    computed_results = {}  # Maps node_id -> result
    
    for node_id in comp_order:
        # Skip analysis nodes (they'll be computed later)
        if node_id[0] == "analysis":
            continue
            
        dep_instance, params = dep_instances[node_id]
        
        # Compute and store the result
        logger.debug(f"Computing dependency: {dep_instance}")
        result = dep_instance.compute(data, **results, **params)
        
        # Store in both dictionaries
        computed_results[node_id] = result
        results[node_id[0]] = result
        
    # Construct dependency results for each analysis
    all_dependency_results = []
    for analysis in analyses_list:
        dependency_results = kwargs.copy()

        for dep_name, dep_source in analysis.dependencies.items():
            if dep_name not in results:
                raise ValueError(f"Dependency '{dep_name}' for {analysis.name} was not computed")

            node_id, _, _ = resolve_dependency_node(analysis, dep_name, dep_source)
            
            if node_id in computed_results:
                dependency_results[dep_name] = computed_results[node_id]
        
        all_dependency_results.append(dependency_results)
    
    return all_dependency_results