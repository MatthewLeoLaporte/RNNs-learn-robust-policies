
import pkgutil
import importlib

from rnns_learn_robust_motor_policies.analysis import modules as analysis_modules_pkg


# TODO: Refactor to avoid repetition with the module traversal that follows;
# also add recursion (i.e. nested subpackages)
def discover_subpackages(package):
    """Return all modules of a package."""
    modules = []
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if is_pkg:
            modules.append(importlib.import_module(name))
    return modules


ANALYSIS_REGISTRY = {}
for subpkg in discover_subpackages(analysis_modules_pkg):
    for _, name, _ in pkgutil.iter_modules(subpkg.__path__, subpkg.__name__ + '.'):
        module = importlib.import_module(name)
        if hasattr(module, 'ID'):
            ANALYSIS_REGISTRY[module.ID] = module
            
