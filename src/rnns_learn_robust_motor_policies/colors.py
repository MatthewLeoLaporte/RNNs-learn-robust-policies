
from collections.abc import Hashable, Sequence
from typing import Literal, Optional

import jax.tree as jt
from jaxtyping import PyTree
import plotly.colors as plc

import feedbax.plotly as fbp


# How much to darken (<1) or lighten (>1) plots of means, versus plots of individual trials
MEAN_LIGHTEN_FACTOR = 0.7


# Colorscales
COLORSCALES = dict(
    reach_condition='phase',
    trials='Tealgrn',
    disturbance_train_stds='viridis',
    disturbance_amplitudes='plotly3',
    fb_pert_vars=plc.qualitative.D3,
    context_inputs='thermal',
)


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

