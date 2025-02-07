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


def build_dependency_graph(analyses: list[Type[AbstractAnalysis]]) -> tuple[dict[str, Set[str]], Set[str]]:
    """Build adjacency list representation of dependency graph."""
    graph = defaultdict(set)
    all_deps = set()

    # Collect all dependencies
    for analysis in analyses:
        for dep_name, dep_class in analysis.dependencies.items():
            graph[dep_class.__name__].add(dep_name)
            all_deps.add(dep_name)

    return dict(graph), all_deps


def topological_sort(graph: dict[str, Set[str]], all_deps: Set[str]) -> list[str]:
    """Return dependencies in the order they should be computed.
    """
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
    for dep in all_deps:
        if dep not in visited:
            visit(dep)

    return order


def compute_dependencies(
    analyses: list[Type[AbstractAnalysis]],
    models: PyTree[Module],
    tasks: PyTree[Module],
    states: PyTree[Module],
    hps: TreeNamespace,
) -> dict:
    """Compute all dependencies in correct order."""
    # Build dependency graph
    graph, all_deps = build_dependency_graph(analyses)

    # Get computation order
    comp_order = topological_sort(graph, all_deps)

    # Map dependency names to their computation classes
    dep_classes = {
        dep_name: dep_class
        for analysis in analyses
        for dep_name, dep_class in analysis.dependencies.items()
    }

    # Compute dependencies in order
    results = {}
    for dep_name in comp_order:
        dep_class = dep_classes[dep_name]
        # Initialize dependency class and compute
        dep_instance = dep_class()
        # Map over pytree of states
        results[dep_name] = dep_instance.compute(models, tasks, states, hps, **results)

    return results