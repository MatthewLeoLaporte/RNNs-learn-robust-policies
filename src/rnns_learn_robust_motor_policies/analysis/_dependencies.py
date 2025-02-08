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

from collections import defaultdict
from typing import Set, Type

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree

from jax_cookbook import is_module

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace


def build_dependency_graph(analyses: list[Type[AbstractAnalysis]]) -> tuple[dict[str, Set[str]], dict]:
    graph = defaultdict(set)
    dep_classes = {}
    
    def add_deps(analysis_or_dep):
        for dep_name, dep_class in analysis_or_dep.dependencies.items():
            # Record the direct mapping
            dep_classes[dep_name] = dep_class
            # Add edges for this dependency's dependencies
            for subdep_name in dep_class.dependencies:
                graph[dep_name].add(subdep_name)
            # Recursion will handle subdependencies
            add_deps(dep_class)
        
    for analysis in analyses:
        add_deps(analysis)
        
    return dict(graph), dep_classes


def topological_sort(graph: dict[str, Set[str]], dep_classes: dict[str, Type]) -> list[str]:
    """Return dependencies in order they should be computed."""
    visited = set()
    temp_marks = set()
    order = []
    
    def visit(node: str):
        if node in temp_marks:
            raise ValueError("Circular dependency detected")
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
    for dep in dep_classes:
        if dep not in visited:
            visit(dep)
            
    return order


def compute_dependencies(
    analyses: list[Type[AbstractAnalysis]],
    models: PyTree[Module],
    tasks: PyTree[Module],
    states: PyTree[Module],
    hps: PyTree[TreeNamespace],
    **kwargs,
) -> dict:
    """Compute all dependencies in correct order.
    
    Any `kwargs` are provided as baseline dependencies, and included in the returned dict.
    """
    # Build dependency graph
    graph, dep_classes = build_dependency_graph(analyses)  # Use both return values
    
    # Get computation order
    comp_order = topological_sort(graph, dep_classes)
    
    # Compute dependencies in order
    results = kwargs
    for dep_name in comp_order:
        dep_class = dep_classes[dep_name]  # Now this will work
        # Initialize dependency class and compute
        dep_instance = dep_class()
        results[dep_name] = dep_instance.compute(models, tasks, states, hps, **results)
        
    return results