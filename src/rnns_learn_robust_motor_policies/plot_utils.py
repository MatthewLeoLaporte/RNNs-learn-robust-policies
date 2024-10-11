
from pathlib import Path
from typing import Optional
import matplotlib.figure as mplfig
import plotly.graph_objects as go


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

        label += suffix
        
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
    
    return savefig