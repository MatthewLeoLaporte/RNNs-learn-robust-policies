from collections.abc import Callable
import equinox as eqx 
from jaxtyping import PyTree
import plotly.graph_objects as go

from feedbax import is_module
from feedbax._tree import eitherf, is_type


def swap_model_trainables(model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )
    

def subdict(dct, keys):
    return {k: dct[k] for k in keys}


def pp(tree):
    """Helper to pretty-print PyTrees, truncating objects commonly treated as leaves during data analysis."""
    eqx.tree_pprint(tree, truncate_leaf=eitherf(is_module, is_type(go.Figure)))