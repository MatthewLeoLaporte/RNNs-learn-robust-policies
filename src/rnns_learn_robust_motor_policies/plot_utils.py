
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from jaxtyping import Array, Float
import jax.numpy as jnp
import matplotlib.figure as mplfig
import plotly.graph_objects as go

from rnns_learn_robust_motor_policies.setup_utils import filename_join


def get_savefig_func(fig_dir: Path, suffix=""):
    """Returns a function that saves Matplotlib and Plotly figures to file in a given directory.
    
    This is convenient in notebooks, where all figures made within a single notebook are generally
    saved to the same directory. 
    """
    
    def savefig(fig, label, ext='.svg', transparent=True, subdir: Optional[str] = None, **kwargs): 

        if subdir is not None:
            save_dir = fig_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True) 
        else:
            save_dir = fig_dir           

        label = filename_join([label, suffix])
        
        if isinstance(fig, mplfig.Figure):
            fig.savefig(
                save_dir / f"{label}{ext}",
                transparent=transparent, 
                **kwargs, 
            )
        
        elif isinstance(fig, go.Figure):
            # Save HTML for easy viewing, and JSON for embedding.
            fig.write_html(save_dir / f'{label}.html')
            fig.write_json(save_dir / f'{label}.json')
            
            # Also save PNG for easy browsing and sharing
            fig.write_image(save_dir / f'{label}.png', scale=2)
            # fig.write_image(save_dir / f'{label}.webp', scale=2)
    
    return savefig


def add_context_annotation(
    fig: go.Figure,
    train_condition_strs: Optional[Sequence[str]] = None, 
    perturbations: Optional[dict[str, tuple[float, Optional[int], Optional[int]]]] = None,
    n=None,
    i_trial=None,
    i_replicate=None,
    i_condition=None,
    y=1.1,
) -> None:
    """Annotates a figure with details about sample size, trials, replicates."""
    lines = []
    if train_condition_strs is not None:
        for condition_str in train_condition_strs:
            lines.append(f"Trained on {condition_str}")
        
    if perturbations is not None:
        for label, (amplitude, start, end) in perturbations.items():
            line = f"Response to amplitude {amplitude} {label} "
            match (start, end):
                case (None, None):
                    line += 'constant over trial'
                case (None, _):
                    line += f'from trial start to step {end}'
                case (_, None):
                    line += f'from step {start} to trial end'
                case (_, _):
                    line += f'from step {start} to {end}'
                
            lines.append(line)
    
    match (n, i_trial, i_replicate):
        case (None, None, None):
            pass
        case (n, None, None):
            lines.append(f"N = {n}")
        case (n, i_trial, None):
            lines.append(f"Single evaluation (#{i_trial}) of N = {n} model replicates")
        case (n, None, i_replicate):
            lines.append(f"N = {n} evaluations of model replicate {i_replicate}")
        case (None, i_trial, i_replicate):
            lines.append(f"Single evaluation (#{i_trial}) of model replicate {i_replicate}")
        case _:
            raise ValueError("Invalid combination of n, i_trial, and i_replicate for annotation")
    
    if i_condition is not None:
        lines.append(f"For single task condition ({i_condition})")
                
    fig.update_layout(margin_t=100 + 5 * len(lines))
    
    fig.add_annotation(dict(
        text='<br>'.join(lines),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=y,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=16),  # Adjust font size as needed
    ))
    

# TODO: Make inits and goals individually optional
# Perhaps by simplifying this function to just plot a single set of points (`points_2D` or something)
# and then just mapping it over the endpoints later, and having a separate function for lines (straight guides)
def add_endpoint_traces(
    fig: go.Figure,
    pos_endpoints: Float[Array, "ends=2 *trials xy=2"],
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
                    mode="markers",
                    marker=kws,
                    marker_colorscale=colorscale,
                    showlegend=True,
                    **kwargs,
                )
            ]
        )
