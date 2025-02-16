
from collections.abc import Callable, Hashable, Sequence
from typing import ClassVar, Literal, Optional

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree
import plotly.colors as plc

import feedbax.plotly as fbp
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace, dict_to_namespace


# How much to darken (<1) or lighten (>1) plots of means, versus plots of individual trials
MEAN_LIGHTEN_FACTOR = 0.7


# Colorscales
#! TODO: Combine with `setup_colors`; pair a colorscale and a variable under a single key
COLORSCALES: dict[str, str] = dict(
    context_input='thermal',
    disturbance_amplitude='plotly3',
    disturbance_std='viridis',
    # pert_var=plc.qualitative.D3,  # list[str]
    reach_condition='phase',
    trial='Tealgrn',
)


def get_color_vars(var_funcs: dict[str, Callable], hps: TreeNamespace) -> dict[str, Optional[Sequence]]:
    def _try_get_var(func, hps):
        try:
            return func(hps)
        except AttributeError:
            return None 
    return {k: _try_get_var(v, hps) for k, v in var_funcs.items()}


def setup_colors(hps: PyTree[TreeNamespace], var_funcs: dict[str, Callable]) -> tuple[dict, dict]:
    """Get all the colorscales we might want for our analyses, given the experiment hyperparameters.
    """
    lighten_factors = dict(normal=1, dark=MEAN_LIGHTEN_FACTOR)
    colors = jt.map(
        lambda hps: dict_to_namespace({
            k: get_colors_dicts(v, COLORSCALES[k], lighten_factor=lighten_factors)
            for k, v in get_color_vars(var_funcs, hps).items()
            if v is not None
        }),
        hps,
        is_leaf=is_type(TreeNamespace),
    )
    # discrete_colors = {
    #     k: get_colors_dicts_from_discrete(v, COLORSCALES[k])
    #     for k, v in dict(
    #         pert_var=pert_var_names,
    #     )
    # }
    discrete_colors = {}
    return colors, discrete_colors


def get_colors_dicts_from_discrete(
    keys: Sequence[Hashable], 
    colors: Sequence[str] | Sequence[tuple], 
    lighten_factor: PyTree[float, 'T'] = [1, MEAN_LIGHTEN_FACTOR], 
    colortype: Literal['rgb', 'tuple'] = 'rgb',
) -> PyTree[dict[Hashable, str | tuple], 'T']:
    def _get_colors(colors, factor):
        colors = fbp.adjust_color_brightness(colors, factor)
        return plc.convert_colors_to_same_type(colors, colortype=colortype)[0]
    
    return jt.map(
        lambda f: dict(zip(keys, _get_colors(colors, f))),
        lighten_factor,
    )
    
    
def get_colors_dicts(
    keys: Sequence[Hashable], 
    colorscale: str | Sequence[str] | Sequence[tuple],
    lighten_factor: PyTree[float, 'T'] = [1, MEAN_LIGHTEN_FACTOR], 
    colortype: Literal['rgb', 'tuple'] = 'rgb',
) -> PyTree[dict[Hashable, str | tuple], 'T']:
    colors = fbp.sample_colorscale_unique(colorscale, len(keys))
    return get_colors_dicts_from_discrete(keys, colors, lighten_factor, colortype)




