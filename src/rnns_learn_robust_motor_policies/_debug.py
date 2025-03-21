
import jax.tree as jt
import plotly.graph_objects as go

from jax_cookbook import is_type

from rnns_learn_robust_motor_policies.tree_utils import pp 


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