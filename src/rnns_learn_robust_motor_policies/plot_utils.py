
import matplotlib.figure as mplfig
import plotly.graph_objects as go


def get_savefig_func(fig_dir, suffix=""):
    """Returns a function that saves Matplotlib and Plotly figures to file in a given directory.
    
    This is convenient in notebooks, where all figures made within a single notebook are generally
    saved to the same directory. 
    """
    
    def savefig(fig, label, ext='.svg', transparent=True, **kwargs): 
        
        label += suffix
        
        if isinstance(fig, mplfig.Figure):
            fig.savefig(
                fig_dir / f"{label}{ext}",
                transparent=transparent, 
                **kwargs, 
            )
        
        elif isinstance(fig, go.Figure):
            # Save HTML for easy viewing, and JSON for embedding.
            fig.write_html(fig_dir / f'{label}.html')
            fig.write_json(fig_dir / f'{label}.json')
    
    return savefig