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

import jax.tree as jt

from rnns_learn_robust_motor_policies.analysis.analysis import (
    AbstractAnalysis, AnalysisInputData, _format_dict_of_params
)


logger = logging.getLogger(__name__)


def param_hash(params: Dict[str, Any]) -> str:
    """Create a hash of parameter values to uniquely identify dependency configurations."""
    # Convert params to a stable string representation and hash it
    params_formatted = _format_dict_of_params(params)
    param_str = json.dumps(params_formatted, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def get_params_for_dep_class(analysis, dep_class):
    """Get parameters for a dependency based on its class."""
    # Check if the class exists in dep_params
    dep_params = getattr(analysis, 'dependency_params', {})
    return dep_params.get(dep_class, {})


def get_dependency_id_and_params(analysis, dep_name, dep_class):
    class_params = analysis.dependency_kwargs().get(dep_name, {})
    field_params = get_params_for_dep_class(analysis, dep_class)
    params = {**field_params, **class_params}
    node_id = f"{dep_name}_{param_hash(params)}"
    return node_id, params


def build_dependency_graph(analyses: Sequence[AbstractAnalysis]) -> tuple[dict[str, Set[str]], dict]:
    """Build dependency graph with parameter-specific nodes.
    
    Each dependency with unique parameters gets a unique node in the graph.
    Only include dependencies for analyses whose conditions are met.
    """
    graph = defaultdict(set)
    dep_instances = {}  # Maps node_id -> (dep_class, params)
    
    # TODO: Filter analyses by their conditions
    # analyses_to_process = [a for a in analyses if conditions_are_met(a)]

    def add_deps(analysis):        
        for dep_name, dep_class in analysis.dependencies.items():
            node_id, params = get_dependency_id_and_params(
                analysis, dep_name, dep_class
            )
            
            if node_id not in dep_instances:
                # Record the mapping
                dep_instances[node_id] = (dep_class, params)
                add_subdeps(dep_class, params, node_id)
            
    
    def add_subdeps(dep_class, params, node_id):
        temp_instance = dep_class(**params)
        for subdep_name, subdep_class in temp_instance.dependencies.items():
            subdep_id, subdep_params = get_dependency_id_and_params(
                temp_instance, subdep_name, subdep_class
            )
            
            graph[node_id].add(subdep_id)
            
            if subdep_id not in dep_instances:
                dep_instances[subdep_id] = (subdep_class, subdep_params)
                add_subdeps(subdep_class, subdep_params, subdep_id)
        
    for analysis in analyses:  
        add_deps(analysis)

    for node_id in dep_instances:
        if node_id not in graph:
            graph[node_id] = set()
        
    return dict(graph), dep_instances


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


def compute_dependencies(
    analyses: Sequence[AbstractAnalysis],
    data: AnalysisInputData,
    **kwargs,
) -> dict:
    """Compute all dependencies in correct order.
    
    Any `kwargs` are provided as baseline dependencies, and included in the returned dict.
    """
    # Build dependency graph with parameter-specific nodes
    graph, dep_instances = build_dependency_graph(analyses)
    
    # Get computation order
    comp_order = topological_sort(graph)
    
    # Compute dependencies in order
    results = kwargs.copy()
    computed_results = {}  # Maps node_id -> result
    
    for node_id in comp_order:
        dep_class, params = dep_instances[node_id]
        
        # Create instance with parameters
        dep_instance = dep_class(**params)
        
        # Extract the base name (without parameter hash)
        base_name = node_id.rsplit('_', 1)[0]
        
        # Compute and store the result, passing all the dependencies computed so far
        logger.debug(f"Computing dependency: {dep_instance}")
        result = dep_instance.compute(data, **results, **params)
        
        # Store in both dictionaries
        computed_results[node_id] = result
        results[base_name] = result
        
    # Construct a dict of dependency results to supply to each of the requested analyses
    all_dependency_results = []
    for analysis in analyses:
        dependency_results = kwargs.copy()

        for dep_name, dep_class in analysis.dependencies.items():
            if dep_name not in results:
                raise ValueError(f"Dependency '{dep_name}' for {analysis.name} was not computed")

            node_id, _ = get_dependency_id_and_params(analysis, dep_name, dep_class)
            
            if node_id in computed_results:
                dependency_results[dep_name] = computed_results[node_id]
        
        all_dependency_results.append(dependency_results)
    
    return all_dependency_results