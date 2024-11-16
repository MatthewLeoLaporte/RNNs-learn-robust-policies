
from collections.abc import Sequence
import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float
import matplotlib.figure as mplfig
import plotly
import plotly.colors as plc
import plotly.graph_objects as go
from IPython.display import HTML, display

from feedbax import is_type
import feedbax.plot as fbp

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
                str(save_dir / f"{label}{ext}"),
                transparent=transparent, 
                **kwargs, 
            )
        
        elif isinstance(fig, go.Figure):
            # Save HTML for easy viewing, and JSON for embedding.
            # fig.write_html(save_dir / f'{label}.html')
            fig.write_json(save_dir / f'{label}.json')
            
            # Also save PNG for easy browsing and sharing
            fig.write_image(save_dir / f'{label}.png', scale=2)
            # fig.write_image(save_dir / f'{label}.webp', scale=2)
    
    return savefig


def figs_flatten_with_paths(figs):
    return jax.tree_util.tree_flatten_with_path(figs, is_leaf=is_type(go.Figure))[0]


def figleaves(tree):
    return jt.leaves(tree, is_leaf=is_type(go.Figure))


def add_context_annotation(
    fig: go.Figure,
    train_condition_strs: Optional[Sequence[str]] = None, 
    perturbations: Optional[dict[str, tuple[float, Optional[int], Optional[int]]]] = None,
    n=None,
    i_trial=None,
    i_replicate=None,
    i_condition=None,
    y=1.1,
    **kwargs,
) -> go.Figure:
    """Annotates a figure with details about sample size, trials, replicates."""
    lines = []
    if train_condition_strs is not None:
        for condition_str in train_condition_strs:
            lines.append(f"Trained on {condition_str}")
        
    if perturbations is not None:
        for label, (amplitude, start, end) in perturbations.items():
            if amplitude is None:
                amplitude_str = ''
            else:
                amplitude_str = f" amplitude {amplitude:.2g}"
                
            line = f"Response to{amplitude_str} {label} "
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
    
    # Adjust the layout of the figure to make room for the annotation
    if (margin_t := fig.layout.margin.t) is None:  # type: ignore
        margin_t = 100
    
    fig.update_layout(margin_t=margin_t + 5 * len(lines))
    
    if (height := fig.layout.height) is not None: # type: ignore    
        fig.update_layout(height=height + 5 * len(lines))  
    
    fig.add_annotation(dict(
        text='<br>'.join(lines),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=y,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=12),  # Adjust font size as needed
        name='context_annotation',
        **kwargs,
    ))
    
    return fig


def get_merged_context_annotation(*figs):
    """Given figures with annotations added by `add_context_annotation`, return the text of a merged annotation.
    
    Note that this does not add the text as an annotation to any figure.
    """
    annotations_text = [
        next(iter(fig.select_annotations(selector=dict(name="context_annotation")))).text 
        for fig in figs
    ]
    annotation_unique_lines = set(sum([text.split('<br>') for text in annotations_text], []))
    merged_annotation = '<br>'.join(reversed(sorted(annotation_unique_lines)))
    return merged_annotation  


def toggle_bounds_visibility(fig):
    """Toggle the visibility of traces with 'bound' in their names."""
    def toggle_visibility_if_bound(trace):
        if 'bound' in trace.name:
            if trace.visible is None:
                trace.visible = False
            else:
                trace.visible = not trace.visible
    
    fig.for_each_trace(toggle_visibility_if_bound)


def plotly_vscode_latex_fix():
    """Fixes LaTeX rendering in Plotly figures in VS Code."""
    if os.environ.get('VSCODE_PID') is not None:        
        plotly.offline.init_notebook_mode()
        display(HTML(
            '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
        ))