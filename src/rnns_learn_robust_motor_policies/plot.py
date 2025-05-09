from collections.abc import Sequence
import math
import re
from typing import Callable, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt 
from jaxtyping import Array, Bool, Float, PyTree 
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
    colorscale: Optional[str] = None,  
    colorscale_axis: int = 0,  # of `trials` axes
    init_marker_kws: Optional[dict] = None, 
    goal_marker_kws: Optional[dict] = None,
    straight_guides: bool = False,
    straight_guide_kws: Optional[dict] = None,
    behind_traces: bool = True,
    **kwargs,
):
    """
    Add endpoint markers and optional straight guide lines to a Plotly figure.
    
    Args:
        fig: Plotly figure to add traces to
        pos_endpoints: Array with start and goal positions. Shape: [2, *trials, 2] where
                      the first dimension is [start, goal], middle dimensions are trials,
                      and last dimension is [x, y] coordinates
        visible: tuple of booleans indicating if start and goal markers should be visible
        colorscale: Optional color scale to apply to the markers
        colorscale_axis: Which trial axis to use for color mapping (if colorscale provided)
        init_marker_kws: Additional keyword arguments for the start markers
        goal_marker_kws: Additional keyword arguments for the goal markers
        straight_guides: Whether to add straight lines connecting start and goal points
        straight_guide_kws: Additional keyword arguments for the guide lines
        **kwargs: Additional keyword arguments for all marker traces
        
    Returns:
        Updated Plotly figure with added traces
    """
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
    default_guide_kws = dict(
        mode="lines",
        line=dict(
            color="rgba(100, 100, 100, 0.5)",
            dash="dot",
            width=0.5,
        ),
        hoverinfo="skip",
    )
     
    if len(pos_endpoints.shape) == 2:
        pos_endpoints = jnp.expand_dims(pos_endpoints, axis=1)
    
    if colorscale is not None:
        # Calculate colors based on trial dimensions only
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
        
        for i, key in enumerate(marker_kws):
            marker_kws[key].update(
                # Apply the flat color array to markers
                line_color=flat_colors, 
                color=flat_colors,
                cmin=0,
                cmax=1,
                # colorscale=colorscale, # This should be set on the trace, not marker
            )

    if init_marker_kws is not None:
        marker_kws["Start"].update(init_marker_kws)
    if goal_marker_kws is not None:
        marker_kws["Goal"].update(goal_marker_kws)
    
    if len(pos_endpoints.shape) > 3:
        pos_endpoints = jnp.reshape(pos_endpoints, (2, -1, 2))
    
    traces = []

    # Add straight guide lines between start and goal points if requested
    if straight_guides:
        if straight_guide_kws is not None:
            default_guide_kws.update(straight_guide_kws)

        for i in range(pos_endpoints.shape[1]):
            # For each trial, create a line from start to goal
            traces.append(go.Scatter(
                x=[pos_endpoints[0, i, 0], pos_endpoints[1, i, 0]],
                y=[pos_endpoints[0, i, 1], pos_endpoints[1, i, 1]],
                name="Straight path",
                showlegend=False,
                legend="legend2",
                **default_guide_kws,
            ))

    for j, (label, kws) in enumerate(marker_kws.items()):
        traces.append(go.Scatter(
            name=f"{label}",
            meta=dict(label=label),
            legend="legend2",
            hovertemplate=f"{label}<extra></extra>",
            x=pos_endpoints[j, ..., 0],
            y=pos_endpoints[j, ..., 1],
            visible=visible[j],
            mode="markers",
            marker=kws,
            marker_colorscale=colorscale,
            showlegend=True,
            **kwargs,
        ))
        
    if behind_traces:
        existing_traces = list(fig.data)
        fig.data = ()
        fig.add_traces(traces + existing_traces)
    else:
        fig.add_traces(traces)

    fig.update_layout(
        legend2=dict(
            title_text="Guides",
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            y=0.99,
            x=1.22,
        ),
    )

    return fig


def get_violins(\
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
    


def plot_eigvals_df(
    df, 
    marginals='box', 
    trace_kws=None, 
    scatter_kws=None, 
    layout_kws=None, 
    marginal_boundary_lines=True,
    **kwargs,
):
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
        render_mode='svg',
        **kwargs,
    )

    if scatter_kws is not None:
        def _update_scatter(trace):
            if isinstance(trace, (go.Scatter, go.Scattergl)):
                trace.update(**scatter_kws)
                
        fig.for_each_trace(_update_scatter)

    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=600,
        height=450,
    )
    fig.add_shape(
        type='circle',
        xref='x', yref='y',
        x0=-1, y0=-1, x1=1, y1=1,
        fillcolor='white',
        layer='below',
        name="boundary_circle",
        **stable_boundary_kws,
    )
    if marginal_boundary_lines:
        for coord in [-1, 1]:
            fig.add_vline(
                x=coord, 
                row=0, 
                name="boundary_line",
                **stable_boundary_kws,
            )
            fig.add_hline(
                y=coord, 
                col=2, 
                name="boundary_line",
                **stable_boundary_kws,
            )
    
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[0, 0],
        mode='lines',
        line_dash='dot',
        line_color='grey',
        line_width=1,
        showlegend=False,
        name="zerolines",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[-1, 1],
        mode='lines',
        line_dash='dot',
        line_color='grey',
        line_width=1,
        showlegend=False,
        name="zerolines",
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



def _calculate_axis_bounds(
    axis: Literal['x', 'y'],
    fig_leaves: list[go.Figure],
    padding_factor: float,
    trace_selector: Callable,
) -> tuple[list[str], dict[str, Optional[tuple[float, float]]]]:
    """
    Calculates the required min/max bounds for a specific axis across figures,
    subplot by subplot, without applying them.

    Returns:
        A tuple containing:
        - list of subplot indices found (e.g., ["", "2", "3"]).
        - dictionary mapping subplot index to the calculated (padded) range
          (min, max) or None if no valid data found.
    """
    if not fig_leaves:
        return [], {}

    # --- Determine subplot indices ---
    layout_keys = fig_leaves[0].to_dict()['layout'].keys()
    axis_prefix = f"{axis}axis"
    subplot_indices: list[str] = []
    if axis_prefix in layout_keys:
        subplot_indices.append("")
    for key in layout_keys:
         match = re.fullmatch(f"{axis_prefix}(\d+)", key)
         if match:
             subplot_indices.append(match.group(1))
    subplot_indices.sort(key=lambda s: int(s) if s else 0)

    if not subplot_indices:
        return [], {}

    # --- Calculate global bounds per subplot ---
    global_bounds: dict[str, dict] = {
        idx: {'min': math.inf, 'max': -math.inf, 'found': False}
        for idx in subplot_indices
    }

    for subplot_index in subplot_indices:
        current_axis_layout_key = f"{axis}axis{subplot_index}"
        for fig in fig_leaves:
            for trace in fig.data:
                if not trace_selector(trace): continue

                trace_axis_assignment_attr = f"{axis}axis"
                trace_target_axis_id = getattr(trace, trace_axis_assignment_attr, None)
                effective_trace_layout_key = f"{axis}axis" # Default
                if trace_target_axis_id is not None:
                    match_id = re.fullmatch(f"{axis}(\d*)", trace_target_axis_id)
                    if match_id:
                         effective_trace_layout_key = f"{axis}axis{match_id.group(1)}"
                    else: continue # Skip trace if axis id format is unexpected

                if effective_trace_layout_key == current_axis_layout_key:
                    coords = getattr(trace, axis, None)
                    numeric_coords_list = []
                    valid_coords_found_in_trace = False
                    if coords is not None and hasattr(coords, '__iter__') and not isinstance(coords, (str, bytes)):
                        for val in coords:
                            if isinstance(val, (int, float, np.number)):
                                numeric_coords_list.append(float(val))
                                valid_coords_found_in_trace = True
                            elif val is None:
                                numeric_coords_list.append(np.nan)
                                valid_coords_found_in_trace = True

                    if valid_coords_found_in_trace:
                        numeric_coords = np.array(numeric_coords_list, dtype=float)
                        if numeric_coords.size > 0:
                            current_min = np.nanmin(numeric_coords)
                            current_max = np.nanmax(numeric_coords)
                            if np.isfinite(current_min) and np.isfinite(current_max):
                                subplot_data = global_bounds[subplot_index]
                                subplot_data['min'] = min(subplot_data['min'], current_min)
                                subplot_data['max'] = max(subplot_data['max'], current_max)
                                subplot_data['found'] = True

    # --- Determine final range tuple per subplot ---
    final_ranges: dict[str, Optional[tuple[float, float]]] = {}
    for subplot_index in subplot_indices:
        subplot_data = global_bounds[subplot_index]
        g_min, g_max, found_data = subplot_data['min'], subplot_data['max'], subplot_data['found']
        subplot_final_range = None
        if found_data and g_min <= g_max:
            non_negative_padding_factor = max(0.0, padding_factor)
            if np.isclose(g_min, g_max):
                padding_abs = abs(g_min) * non_negative_padding_factor if not np.isclose(g_min, 0.0) else 0.5 * non_negative_padding_factor
                if non_negative_padding_factor > 0 and np.isclose(padding_abs, 0.0) and not np.isclose(g_min, 0.0):
                    padding_abs = 0.5 * non_negative_padding_factor
                padded_min = g_min - padding_abs
                padded_max = g_max + padding_abs
                if padded_min > padded_max: padded_min = padded_max = g_min
            else:
                delta = g_max - g_min
                padding = delta * non_negative_padding_factor
                padded_min = g_min - padding
                padded_max = g_max + padding
            subplot_final_range = (padded_min, padded_max)
        final_ranges[subplot_index] = subplot_final_range

    return subplot_indices, final_ranges


# --- Standalone function to set single axis ---
def set_axis_bounds_equal(
    axis: Literal['x', 'y'],
    figs: PyTree,
    padding_factor: float = 0.05,
    trace_selector: Callable = lambda trace: True,
    **kwargs,
) -> PyTree:
    """
    Finds and applies global bounds for a *single specified axis* ('x' or 'y')
    across all figures in a PyTree, subplot by subplot.
    NOTE: This function applies bounds directly and does *not* account for
    `scaleanchor` interactions if you later modify the other axis. For
    synchronized scaling, use `set_axes_bounds_equal`.

    Args:
        figs: A PyTree containing go.Figure objects.
        axis: The axis ('x' or 'y') to synchronize.
        padding_factor: Padding factor for the range calculation.
        trace_selector: Function to select traces for bounds calculation.

    Returns:
        PyTree with the specified axis updated subplot-wise.
    """
    # Corrected: Use jt.leaves and jt.map
    leaves = jt.leaves(figs)
    if not leaves: return figs
    fig_leaves = [leaf for leaf in leaves if isinstance(leaf, go.Figure)]
    if not fig_leaves: return figs

    subplot_indices, final_ranges = _calculate_axis_bounds(
        axis, fig_leaves, padding_factor, trace_selector
    )

    if not final_ranges: return figs

    def _update_leaf_single_axis(leaf):
        if isinstance(leaf, go.Figure):
            for subplot_index, final_range in final_ranges.items():
                if final_range is not None:
                    axis_attr_name = f"{axis}axis{subplot_index}"
                    axis_obj = getattr(leaf.layout, axis_attr_name, None)
                    if axis_obj:
                        axis_obj.range = final_range
        return leaf
    # Corrected: Use jt.map
    return jt.map(_update_leaf_single_axis, figs)


#! On preliminary tests this isn't actually very useful for anchored axes;
#! in principle what we need to do for anchored axes is convert x and y 
#! extrema into a common frame (based on the scale ratio) and then scale 
#! that frame (i.e. only x xor y axis range). Does this achieve that?
def set_axes_bounds_equal(
    figs: PyTree,
    padding_factor: float = 0.05,
    trace_selector: Callable = lambda trace: True,
) -> PyTree:
    """
    Synchronizes both 'x' and 'y' axes across all figures in a PyTree,
    subplot by subplot, accounting for `scaleanchor` constraints.

    Calculates required bounds for x and y, then determines the final ranges
    to set on one or both axes per subplot to ensure all data is visible
    while respecting any `scaleanchor` and `scaleratio` properties.

    Args:
        figs: A PyTree containing go.Figure objects.
        padding_factor: Padding factor for the range calculation.
        trace_selector: Function to select traces for bounds calculation.

    Returns:
        PyTree with both x and y axes synchronized subplot-wise, respecting scaling.
    """
    # Corrected: Use jt.leaves and jt.map
    leaves = jt.leaves(figs)
    if not leaves: return figs
    fig_leaves = [leaf for leaf in leaves if isinstance(leaf, go.Figure)]
    if not fig_leaves: return figs

    # Calculate desired ranges for both axes independently
    x_indices, final_x_ranges = _calculate_axis_bounds('x', fig_leaves, padding_factor, trace_selector)
    y_indices, final_y_ranges = _calculate_axis_bounds('y', fig_leaves, padding_factor, trace_selector)

    all_indices = sorted(list(set(x_indices) | set(y_indices)), key=lambda s: int(s) if s else 0)
    if not all_indices: return figs # No axes found

    # Get scale anchor info from the first figure (assuming consistency)
    scale_info = {}
    first_fig_layout = fig_leaves[0].layout
    for idx in all_indices:
        xn, yn = f"xaxis{idx}", f"yaxis{idx}"
        x_axis_obj = getattr(first_fig_layout, xn, None)
        y_axis_obj = getattr(first_fig_layout, yn, None)

        # Check if y axis anchors to x axis for this subplot
        if y_axis_obj and getattr(y_axis_obj, 'scaleanchor', None) == f"x{idx}":
            scale_info[idx] = {'axis': 'y', 'anchor_to': 'x', 'ratio': getattr(y_axis_obj, 'scaleratio', 1.0)}
        # Check if x axis anchors to y axis for this subplot
        elif x_axis_obj and getattr(x_axis_obj, 'scaleanchor', None) == f"y{idx}":
            scale_info[idx] = {'axis': 'x', 'anchor_to': 'y', 'ratio': getattr(x_axis_obj, 'scaleratio', 1.0)}

    # Determine final ranges to actually set for each subplot
    ranges_to_set = {} # {subplot_idx: {'x': range | None, 'y': range | None}}
    for idx in all_indices:
        x_range = final_x_ranges.get(idx)
        y_range = final_y_ranges.get(idx)
        scaling = scale_info.get(idx)

        current_x_final = x_range # Default: set independently
        current_y_final = y_range # Default: set independently

        # If scaling constraint exists and both ranges were calculated
        if scaling and x_range and y_range:
            s = scaling['ratio']
            if s is None:
                s = 1.0
            # Calculate required spans (ensure non-negative)
            Sx = max(0.0, x_range[1] - x_range[0])
            Sy = max(0.0, y_range[1] - y_range[0])

            # Check for valid scaleratio before proceeding
            if s > 1e-9: # Avoid division by zero or near-zero ratio issues
                if scaling['axis'] == 'y': # yaxis anchors to xaxis (y = s*x)
                    # Calculate the span the x-axis MUST have to accommodate the y-span via scaling
                    required_Sx_for_y = Sy / s
                    # The final x-span is the max needed for its own data or for y's data via scaling
                    final_Sx = max(Sx, required_Sx_for_y)
                    # Center the final x-range around the original x-data midpoint
                    x_center = (x_range[0] + x_range[1]) / 2
                    current_x_final = (x_center - final_Sx / 2, x_center + final_Sx / 2)
                    current_y_final = None # Let Plotly determine y-range based on x and ratio

                elif scaling['axis'] == 'x': # xaxis anchors to yaxis (x = s*y)
                    # Calculate the span the y-axis MUST have to accommodate the x-span via scaling
                    required_Sy_for_x = Sx / s
                    # The final y-span is the max needed for its own data or for x's data via scaling
                    final_Sy = max(Sy, required_Sy_for_x)
                     # Center the final y-range around the original y-data midpoint
                    y_center = (y_range[0] + y_range[1]) / 2
                    current_y_final = (y_center - final_Sy / 2, y_center + final_Sy / 2)
                    current_x_final = None # Let Plotly determine x-range based on y and ratio

            # else: If scaleratio is invalid (<=0), keep default independent ranges

        ranges_to_set[idx] = {'x': current_x_final, 'y': current_y_final}

    # Apply the determined ranges using jt.map
    def _update_leaf_final_axes(leaf):
        if isinstance(leaf, go.Figure):
            for idx, axis_ranges in ranges_to_set.items():
                x_range_to_set = axis_ranges['x']
                y_range_to_set = axis_ranges['y']
                xn, yn = f"xaxis{idx}", f"yaxis{idx}"

                # Apply x range if it needs to be set and axis exists
                if x_range_to_set is not None:
                    x_axis_obj_leaf = getattr(leaf.layout, xn, None)
                    if x_axis_obj_leaf:
                        x_axis_obj_leaf.range = x_range_to_set

                # Apply y range if it needs to be set and axis exists
                if y_range_to_set is not None:
                    y_axis_obj_leaf = getattr(leaf.layout, yn, None)
                    if y_axis_obj_leaf:
                        y_axis_obj_leaf.range = y_range_to_set
        return leaf
    # Corrected: Use jt.map
    return jt.map(_update_leaf_final_axes, figs)