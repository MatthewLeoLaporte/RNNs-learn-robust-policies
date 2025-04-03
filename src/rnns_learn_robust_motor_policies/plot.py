from collections.abc import Sequence
from typing import Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt 
from jaxtyping import Array, Bool, Float 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import feedbax.plotly as fbp

from rnns_learn_robust_motor_policies.colors import COLORSCALES
from rnns_learn_robust_motor_policies.types import Responses


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
        # --- Calculate colors based on trial dimensions only ---
        trial_shape = pos_endpoints.shape[1:-1]
        if not trial_shape:  # Handle case with only one trial dimension
             trial_shape = (pos_endpoints.shape[1],)
             
        color_linspace = jnp.linspace(
            0, 1, pos_endpoints.shape[colorscale_axis + 1], endpoint=False
        )
        
        # Axes within trial_shape that are *not* the colorscale_axis
        expand_dims_axes = tuple(i for i in range(len(trial_shape)) if i != colorscale_axis)
        
        # Expand and broadcast linspace to match the full trial shape
        broadcasted_colors = jnp.broadcast_to(
            jnp.expand_dims(color_linspace, expand_dims_axes),
            trial_shape,
        )
        
        # Flatten colors to match the flattened trial dimension used later
        flat_colors = jnp.reshape(broadcasted_colors, (-1,))
        # --- End color calculation ---
        
        for i, key in enumerate(marker_kws):
            marker_kws[key].update(
                # Apply the flat color array to markers
                line_color=flat_colors, 
                color=flat_colors,
                cmin=0,
                cmax=1,
                # colorscale=colorscale, # This should be set on the trace, not marker
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
    
    return fig


def get_violins(
    data: dict[float, dict[float, Float[Array, "..."]]],  # "evals replicates conditions"
    data_split: Optional[dict[float, dict[float, Float[Array, "..."]]]] = None,
    split_mode: Literal['whole', 'split'] = 'whole',
    legend_title: str = "",
    violinmode: Literal['overlay', 'group'] = 'overlay',
    layout_kws: Optional[dict] = None,
    trace_kws: Optional[dict] = None,
    trace_split_kws: Optional[dict] = None,
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
                title_font_size=12,
                range=[0, None],
            ),
            xaxis=dict(
                title=xaxis_title, 
                title_font_size=12,
                type='category',
                range=[-0.75, n_violins - 0.25],
                # tickmode='array',
                tickvals=np.arange(n_violins),
                ticktext=[f'{x:.2g}' for x in example_legendgroup],
            ),
            violinmode=violinmode,
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
        
        
        trace = go.Violin(
            x=xs,
            y=jnp.stack(tuple(data_i.values())).flatten(),
            name=legendgroup_value,
            legendgroup=legendgroup_value,
            scalegroup=legendgroup_value,
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
        
        if data_split is not None:           
            data_split_i = data_split[legendgroup_value]
            
            trace_split = go.Violin(
                x=xs,
                y=jnp.stack(tuple(data_split_i.values())).flatten(),
                name=legendgroup_value,
                legendgroup=legendgroup_value,
                scalegroup=legendgroup_value,
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
            
            if split_mode == 'split':   
                trace.update(side='positive')
                trace_split.update(side='negative')
            elif split_mode == 'whole':
                pass

            if trace_split_kws is not None:
                trace_split.update(**trace_split_kws)
            
            fig.add_trace(trace_split)
            
        if trace_kws is not None:
            trace.update(**trace_kws)
        
        fig.add_trace(trace)
        
        
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
    stable_boundary_kws = dict(
        line=dict(
            color='black',
            width=2,
        ),  
    )
    
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
        fillcolor='white',
        layer='below',
        **stable_boundary_kws,
    )
    for coord in [-1, 1]:
        fig.add_vline(
            x=coord, 
            row=0, 
            **stable_boundary_kws,
        )
        fig.add_hline(
            y=coord, 
            col=2, 
            **stable_boundary_kws,
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


# def plot_fp_loss(all_fps, fp_loss_func, n_bins=50):
#     fp_tols = list(all_fps.keys())
    
#     f1, ax = plt.subplots(figsize=(12, 6))
#     for tol in fp_tols: 
#         ax.semilogy(all_fps[tol]['losses']); 
#         ax.set_xlabel('Fixed point #')
#         ax.set_ylabel('Fixed point loss');
#     f1.legend(fp_tols)
#     ax.set_title('Fixed point loss by fixed point (sorted) and stop tolerance')

#     f2, axs = plt.subplots(1, len(fp_tols), figsize=(12,4))
    
#     for i, tol in enumerate(fp_tols):
#         axs[i].hist(np.log10(fp_loss_func(all_fps[tol]['fps'])), n_bins)
#         axs[i].set_xlabel('log10(FP loss)')
#         axs[i].set_title('Tolerance: ' + str(tol))
    
#     return f1, f2 


def plot_fp_pcs(
    fps_pc: Float[Array, "condition *fp pc"], 
    colors: str | Sequence[str] | None = None, 
    candidates_alpha: float = 0.05, 
    marker_size: int = 3, 
    marker_symbol: str = 'circle',
    #! Is this why we can't plot all the FPs in nb6?
    n_plot_max: int = 1000, 
    candidates_pc: Optional[Float[Array, "candidate pc"]] = None, 
    label: str = 'Fixed points',
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    if candidates_pc is not None:
        emax = candidates_pc.shape[0] if candidates_pc.shape[0] < n_plot_max else n_plot_max
    else:
        emax = n_plot_max

    n_fps_per_condition = np.prod(fps_pc.shape[:-1])
    fps_flat_pc = np.reshape(fps_pc, (-1, fps_pc.shape[-1]))
    
    if isinstance(colors, Sequence):
        colors = np.repeat(colors, n_fps_per_condition)
    
    if fig is None:
        fig = go.Figure(
            layout=dict(
                width=1000, 
                height=1000,
                # title='Fixed point structure and fixed point candidate starting points',
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                ),
            )
        )
    
    if fig is not None:  
        fig.add_trace(
            go.Scatter3d(
                x=fps_flat_pc[0:emax, 0], 
                y=fps_flat_pc[0:emax, 1], 
                z=fps_flat_pc[0:emax, 2], 
                mode='markers',
                marker_color=colors,
                marker_colorscale='phase',
                marker_size=marker_size,
                marker_symbol=marker_symbol,
                name=label,
            ),
        )
        
        if candidates_pc is not None:        
            fig.add_traces(
                [
                    go.Scatter3d(
                        x=candidates_pc[0:emax, 0], 
                        y=candidates_pc[0:emax, 1], 
                        z=candidates_pc[0:emax, 2],
                        mode='markers',
                        marker_size=marker_size,
                        marker_color=f'rgba(0,0,0,{candidates_alpha})',
                        marker_symbol='circle-open',
                        marker_line_width=2,
                        marker_line_color=f'rgba(0,0,0,{candidates_alpha})',
                        name=f'Candidates{label}',
                    ),
                ]
            )
            
            # Lines connecting candidates to respective FPs
            fig.add_traces(
                [
                    go.Scatter3d(
                        x=[candidates_pc[eidx, 0], fps_flat_pc[eidx, 0]], 
                        y=[candidates_pc[eidx, 1], fps_flat_pc[eidx, 1]],
                        z=[candidates_pc[eidx, 2], fps_flat_pc[eidx, 2]], 
                        mode='lines',
                        line_color=f'rgba(0,0,0,{candidates_alpha})',
                        showlegend=False,
                    )
                    for eidx in range(emax)
                ]
            )

    return fig


def plot_traj_and_fp_pcs_3D(
    trajs: Float[Array, "trial time state"], 
    fps: Float[Array, "fp state"], 
    pca: PCA,  # transforms from "state" -> "3"
    colors: str | Sequence[str] | None = None, 
    colors_fps: str | Sequence[str] | None = None, 
    fig: Optional[go.Figure] = None,
):
    if fig is None:
        fig = go.Figure(layout=dict(width=1000, height=1000))
    if colors_fps is None:
        colors_fps = colors
    
    fig = plot_fp_pcs(fps, pca, colors_fps, fig=fig)
    trajs_pcs = pca.transform(
        np.array(trajs).reshape(-1, trajs.shape[-1])
    ).reshape(*trajs.shape[:-1], pca.n_components)  # type: ignore
    fig = fbp.trajectories_3D(trajs_pcs, colors=colors, fig=fig)
    
    return fig


#! TODO: Not sure this should be here. It also redundant with 
#! `analysis.aligned.GET_VARS_TO_ALIGN` except for the origin subtraction
PLANT_VAR_LABELS = Responses('Pos.', 'Vel.', 'Force')
WHERE_PLOT_PLANT_VARS = lambda states: Responses(
    states.mechanics.effector.pos,
    states.mechanics.effector.vel,
    states.efferent.output,
)


def plot_2d_effector_trajectories(
    states, 
    *args, 
    # Corresponding to axis 0 of `states`:
    legend_title='Reach direction', 
    colorscale_key='reach_condition', 
    **kwargs,
):
    """Helper to define the usual formatting for effector trajectory plots."""
    return fbp.trajectories_2D(
        WHERE_PLOT_PLANT_VARS(states),
        var_labels=PLANT_VAR_LABELS,
        axes_labels=('x', 'y'),
        #! TODO: Replace with `colorscales` (common analysis dependency)
        colorscale=COLORSCALES[colorscale_key],
        legend_title=legend_title,
        # scatter_kws=dict(line_width=0.5),
        layout_kws=dict(
            width=100 + len(PLANT_VAR_LABELS) * 300,
            height=400,
            legend_tracegroupgap=1,
        ),
        *args,
        **kwargs,
    )
