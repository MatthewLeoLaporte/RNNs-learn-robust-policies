
from collections.abc import Callable, Hashable, Sequence
from typing import ClassVar, Literal, Optional

import jax.tree as jt
from jax_cookbook import is_type
from jaxtyping import PyTree
import plotly.colors as plc

import feedbax.plotly as fbp

from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace


# How much to darken (<1) or lighten (>1) plots of means, versus plots of individual trials
MEAN_LIGHTEN_FACTOR = 0.7


# Colorscales
COLORSCALES = dict(
    reach_condition='phase',
    trial='Tealgrn',
    disturbance_std='viridis',
    disturbance_amplitude='plotly3',
    pert_var=plc.qualitative.D3,
    context_input='thermal',
)


# class Colors(AbstractAnalysis):
#     dependencies: ClassVar[dict[str, Callable]] = dict()
#     conditions: ClassVar[tuple[str, ...]] = ()      
    
#     def compute(self, models, tasks, states, hps, **kwargs):
#         #! TODO: 
#         pass
    

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


def setup_colors(hps):
    colors = jt.map(
        lambda hps: {
            k: get_colors_dicts(v, COLORSCALES[k])
            for k, v in dict(
                trial=range(hps.eval_n),
                disturbance_std=hps.load.disturbance.std,
                disturbance_amplitude=hps.disturbance.amplitude,
            ).items()
        },
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

