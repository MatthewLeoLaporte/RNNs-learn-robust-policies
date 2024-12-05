from collections.abc import Sequence
from typing import Optional, TypeVar

import equinox as eqx
from feedbax import is_type
import jax.numpy as jnp
import jax.tree as jt 
from jaxtyping import Array, Bool, Float 
import numpy as np
import plotly.express as px
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
                line_color=color_values[i], 
                color=color_values[i],
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


def get_violins(
    data: dict[float, dict[float, Float[Array, "..."]]],  # "evals replicates conditions"
    legend_title: str = "",
    layout_kws: dict = None,
    arr_axis_labels: Optional[Sequence[str]] = None,  # ["Evaluation", "Replicate", "Condition"]
    zero_hline: bool = False,
    *,
    yaxis_title: str, 
    xaxis_title: str,
    colors: dict[float, str],
):
    """
    Arguments:
        data: Outer dict gives legend groups, inner dict gives x-axis values.
        arr_axis_labels: Indices for array axes are included for outliers,
            for example so the batch/replicate can be identified. These strings
            will be used to label indices into axes of arrays of `data`.
    """
    example_legendgroup = list(data.values())[0]
    n_violins = len(example_legendgroup)
    
    example_arr = jt.leaves(data)[0]
    n_dist = np.prod(example_arr.shape)

    # Construct data for hoverinfo
    customdata = jnp.tile(
        jnp.stack(jnp.unravel_index(
            jnp.arange(n_dist), 
            example_arr.shape,
        ), axis=-1), 
        (len(data), 1),
    ).T
    
    if arr_axis_labels is None:
        arr_axis_labels = [f"dim{i}" for i in range(len(customdata))]
    
    customdata_hovertemplate_strs = [
        f"{label}: %{{customdata[{i}]}}" for i, label in enumerate(arr_axis_labels)
    ]

    fig = go.Figure(
        layout=dict(
            # title=(f"Response to amplitude {legendgroup_value} field <br>N = {n_dist}"),
            width=500,
            height=300,
            legend=dict(
                title=legend_title, 
                title_font_size=12,
                tracegroupgap=1,
            ),
            yaxis=dict(
                title=yaxis_title,
                titlefont_size=12,
                range=[0, None],
            ),
            xaxis=dict(
                title=xaxis_title, 
                titlefont_size=12,
                type='category',
                range=[-0.75, n_violins - 0.25],
                # tickmode='array',
                tickvals=np.arange(n_violins),
                ticktext=[f'{x:.2g}' for x in example_legendgroup],
            ),
            violinmode='overlay',
            violingap=0,
            violingroupgap=0,
            margin_t=60,
        )
    )
    
    if zero_hline:
        fig.add_hline(0, line_dash='dot', line_color='grey')
    
    for i, legendgroup_value in enumerate(data):
        data_i = data[legendgroup_value]
        
        xs = jnp.stack([
            jnp.full_like(data, j)
            for j, data in enumerate(data_i.values())
        ]).flatten()
        
        fig.add_trace(
            go.Violin(
                x=xs,
                y=jnp.stack(tuple(data_i.values())).flatten(),
                name=legendgroup_value,
                legendgroup=legendgroup_value,
                box_visible=False,
                meanline_visible=True,
                line_color=colors[legendgroup_value],
                # showlegend=False,
                opacity=1,
                spanmode='hard',
                scalemode='width',
                # width=1.5,
                customdata=customdata.T,
                hovertemplate='<br>'.join([
                    "%{y:.2f}",
                    *customdata_hovertemplate_strs,
                    "<extra></extra>",                    
                ])
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
    


def plot_eigvals_df(df, marginals='box', trace_kws=None, layout_kws=None, **kwargs):
    fig = px.scatter(
        df,
        x='real',
        y='imag',
        marginal_x=marginals,
        marginal_y=marginals,
        **kwargs,
    )

    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=600,
        height=500,
    )
    fig.add_shape(
        type='circle',
        xref='x', yref='y',
        x0=-1, y0=-1, x1=1, y1=1,
        line_color='black',
    )
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0],
        mode='lines',
        line_dash='dot',
        line_color='grey',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1, 1],
        mode='lines',
        line_dash='dot',
        line_color='grey',
        showlegend=False,
    ))
    
    if trace_kws is not None:
        fig.update_traces(**trace_kws)
        
    if layout_kws is not None:
        fig.update_layout(**layout_kws)
    
    return fig