
import jax.tree as jt
import plotly.graph_objects as go

from jax_cookbook import is_type, is_module, is_none

from rlrmp.tree_utils import (
    pp2 as pp, 
    tree_level_labels as tll, 
    first as fs, 
    first_shape as fsh,
) 





def lf(tree, type_=None):
    if type_ is not None:
        is_leaf = is_type(type_)
    else: 
        is_leaf = None
    leaves = jt.leaves(tree, is_leaf=is_leaf)
    if not leaves:
        return None
    else: 
        return leaves[0] 
    

def lff(tree):
    return lf(tree, is_type(go.Figure))

