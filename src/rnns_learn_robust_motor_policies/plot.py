from collections.abc import Sequence
from typing import Optional

from feedbax import is_type
import jax.numpy as jnp
import jax.tree as jt 
from jaxtyping import Array, Bool, Float 
import numpy as np
import plotly.graph_objects as go

from rnns_learn_robust_motor_policies.plot_utils import (
    add_context_annotation,
)
from rnns_learn_robust_motor_policies.types import PertAmpDict, TrainStdDict


def add_endpoint_traces(
    fig: go.Figure,
    pos_endpoints: Float[Array, "ends=2 *trials xy=2"],
    visible: tuple[bool, bool] = (True, True),
    colorscale: Optional[str] = None,  # overrides `color` properties in `marker_kws` args
    colorscale_axis: int = 0,  # of `trials` axes
    init_marker_kws: Optional[dict] = None, 
    goal_marker_kws: Optional[dict] = None,
    straight_guides: bool = False,
    straight_guide_kws: Optional[dict] = None,
    **kwargs,
):
    marker_kws = {
        "Start": dict(
            size=10,
            symbol='square-open',
            color='rgb(25, 25, 25)',
            line=dict(width=2, color='rgb(25, 25, 25)'),
        ),
        "Goal": dict(
            size=10,
            symbol='circle-open',
            color='rgb(25, 25, 25)',
            line=dict(width=2, color='rgb(25, 25, 25)'),
        ),
    }

    if init_marker_kws is not None:
        marker_kws["Start"].update(init_marker_kws)
    if goal_marker_kws is not None:
        marker_kws["Goal"].update(goal_marker_kws)
     
    if len(pos_endpoints.shape) == 2:
        pos_endpoints = jnp.expand_dims(pos_endpoints, axis=1)
    
    if colorscale is not None:
        constant_color_axes = (0,) + tuple(i for i in range(len(pos_endpoints.shape[1:-1])) if i != colorscale_axis)
        color_values = jnp.reshape(
            jnp.broadcast_to(
                jnp.expand_dims(
                    # jnp.ones(pos_endpoints.shape[colorscale_axis + 1]),
                    jnp.linspace(0, 1, pos_endpoints.shape[colorscale_axis + 1], endpoint=False),
                    constant_color_axes, 
                ),
                pos_endpoints.shape[:-1],
            ),
            (2, -1),
        )
        for i, key in enumerate(marker_kws):
            marker_kws[key].update(
                line_color=color_values[i].item(), 
                color=color_values[i].item(),
                cmin=0,
                cmax=1,
            )
    
    if len(pos_endpoints.shape) > 3:
        pos_endpoints = jnp.reshape(pos_endpoints, (2, -1, 2))
    
    for j, (label, kws) in enumerate(marker_kws.items()):
        fig.add_traces(
            [
                go.Scatter(
                    name=f"{label}",
                    meta=dict(label=label),
                    legendgroup=label,
                    hovertemplate=f"{label}<extra></extra>",
                    x=pos_endpoints[j, ..., 0],
                    y=pos_endpoints[j, ..., 1],
                    visible=visible[j],
                    mode="markers",
                    marker=kws,
                    marker_colorscale=colorscale,
                    showlegend=True,
                    **kwargs,
                )
            ]
        )
        
# TODO: Generalize this; it should work with any two-level nested dict structure, not just pert amp and train std
def get_violins_across_train_conditions(
    measure_data: PertAmpDict[float, TrainStdDict[float, Float[Array, "evals replicates conditions"]]], 
    measure_name: str, 
    colors: dict[float, str],
    layout_kws=None,
):
    example_traindict = jt.leaves(measure_data, is_leaf=is_type(TrainStdDict))[0]
    n_train_std = len(example_traindict)
    n_dist = np.prod(jt.leaves(measure_data)[0].shape)

    fig = go.Figure(
        layout=dict(
            # title=(f"Response to amplitude {disturbance_amplitude} field <br>N = {n_dist}"),
            width=500,
            height=300,
            legend=dict(
                title="Field<br>amplitude",
                title_font_size=12,
                tracegroupgap=1,
            ),
            yaxis=dict(
                title=measure_name,
                titlefont_size=12,
                range=[0, None],
            ),
            xaxis=dict(
                title="Train field std.",
                titlefont_size=12,
                type='category',
                range=[-0.75, n_train_std - 0.25],
                # tickmode='array',
                tickvals=np.arange(n_train_std),
                ticktext=[f'{std:.2g}' for std in example_traindict],
            ),
            violinmode='overlay',
            violingap=0,
            violingroupgap=0,
            margin_t=60,
        )
    )
    
    for i, disturbance_amplitude in enumerate(measure_data):
        measure_data_i = measure_data[disturbance_amplitude]
        
        xs = jnp.stack([
            jnp.full_like(data, j)
            for j, data in enumerate(measure_data_i.values())
        ]).flatten()
        
        fig.add_trace(
            go.Violin(
                x=xs,
                y=jnp.stack(tuple(measure_data_i.values())).flatten(),
                name=disturbance_amplitude,
                legendgroup=disturbance_amplitude,
                box_visible=False,
                meanline_visible=True,
                line_color=colors[disturbance_amplitude],
                # showlegend=False,
                opacity=1,
                spanmode='hard',
                scalemode='width',
                # width=1.5,
            )
        )
        
    if layout_kws is not None:
        fig.update_layout(**layout_kws)
    
    return fig


# TODO
# TODO: annotate types
# TODO
def get_measure_replicate_comparisons(
    data, 
    measure_name: str, 
    colors: dict[float, str],
    included_replicates: Optional[Bool[Array, 'replicates']] = None,
):  
    labels = data.keys()
    data = jnp.stack(list(data.values()))
    
    # Exclude replicates which were excluded from analysis for either training condition
    if included_replicates is not None:
        data = jnp.take(data, included_replicates, axis=-2)

    fig = go.Figure()
    # x axis: replicates
    for i in range(data.shape[-2]):        
        # split violin: smallest vs. largest train disturbance std
        for j, train_std in enumerate(labels):

            data_j = data[j, :, i].flatten()
            
            fig.add_trace(
                go.Violin(
                    x=np.full_like(data_j, i),
                    y=data_j.flatten(),
                    name=train_std,
                    legendgroup=train_std,
                    box_visible=False,
                    meanline_visible=True,
                    line_color=colors[train_std],
                    side='positive' if j == 1 else 'negative',
                    showlegend=(i == 0),
                    spanmode='hard',
                )
            )
    fig.update_layout(
        xaxis_title="Model replicate",
        yaxis_title=measure_name,
        xaxis_range=[-0.5, data.shape[-2] - 0.5],
        xaxis_tickvals=list(range(data.shape[-2])),
        yaxis_range=[0, None],
        violinmode='overlay',
        violingap=0,
        violingroupgap=0,
    )
    
    return fig
    
