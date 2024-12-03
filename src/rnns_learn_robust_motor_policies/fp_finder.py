"""Facilities for finding fixed points of dynamical systems.

Derived from the JAX [implementation](https://github.com/google-research/computation-thru-dynamics/blob/94b47aff4cd029aa9df3d0402047110bc3f3519d/fixed_point_finder/fixed_points.py) 
of Fixed Point Finder, by David Sussillo et al.
"""

from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Literal, Optional

import equinox as eqx
from equinox import Module
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp 
import jax.random as jr 
import jax.tree as jt
from jaxtyping import Array, Bool, Float, PRNGKeyArray, PyTree
from matplotlib import pyplot as plt
import numpy as np
import optax
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from feedbax import is_module
import feedbax.plotly as fbp

# class FixedPoint(Module):
#     value: Array


logger = logging.getLogger(__name__)


class FPFilteredResults(Module):
    fps: Float[Array, "candidates state"]
    losses: Float[Array, "candidates"]
    masks: dict[str, Bool[Array, "candidates"]]
    counts: dict[str, int]

    
class FixedPointFinder(Module):
    """Transformable version of FixedPointFinder, without verbosity.
    
    Arguments:
        func: Callable describing the dynamical system. Takes a state array, 
            and returns an updated state array.
        candidates: Array of initial candidate states. Each will be optimized 
            to converge on a fixed point. 
        tol: Tolerance for the optimization. If the mean loss across candidates
            falls below this value, the optimization will terminate.
        n_batches: Number of batches to run the optimization for, if the 
            tolerance is not reached.
        n_batches_per_iter: Number of optimization iterations to run, between 
            each check of the termination conditions. Note that if `n_batches`
            is not divisible by `n_batches_per_iter`, the optimization may 
            run for up to `n_batches + n_batches_per_iter - 1` (?) iterations,
            if the tolerance is not reached by the penultimate check.
        key: Random key for evaluating `func`. (Not implemented yet!)
    """
    optimizer: optax.GradientTransformation
    
    @eqx.filter_jit
    def __call__(
        self, 
        func: Callable[[Float[Array, "state"]], Float[Array, "state"]], 
        candidates: Float[Array, "n_candidates state"], 
        loss_tol: Float[Array, "1"], 
        n_batches: int = 10_000, 
        n_batches_per_iter: int = 200,
        # key: PRNGKeyArray,  # TODO
    ) -> tuple[int, Float[Array, "n_candidates state"], Float[Array, "n_candidates"]]:
        # NOTE: Like the original Fixed Point Finder, this implementation
        #   1. Optimizes the mean loss of a batch of candidates simultaneously,
        #      rather than optimizing each candidate individually. This seems to be
        #      necessary for good convergence.
        #   2. Runs multiple iterations of the optimization before each check of the
        #      termination criteria; this shouldn't strictly be necessary but its
        #      results are consistent with the original implementation.
        
        # Batch the function since we're optimizing multiple candidates at once.
        batch_func = jax.vmap(func, in_axes=(0,))
        # This loss takes the mean over all the candidates.
        loss_func = get_total_fp_loss_func(func)
        opt_state = self.optimizer.init(candidates)
        # The candidate points themselves are the parameters to be optimized; 
        # over the optimization they should converge on fixed points.
        params = candidates
        
        val = (0, params, jnp.inf, opt_state)
        
        def updates(params, opt_state):
            def update(batch_idx, result):
                params, opt_state, loss = result
                
                loss, grads = jax.value_and_grad(loss_func)(params)
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                result = params, opt_state, loss 
                return result
            
            result = params, opt_state, jnp.inf
            return jax.lax.fori_loop(0, n_batches_per_iter, update, result)
        
        def cond_func(val):
            i_batch, _, loss, _ = val
            return (i_batch < n_batches) & (loss > loss_tol)
        
        def body_func(val):
            i_batch, params, _, opt_state = val
            params, opt_state, loss = updates(params, opt_state)
            i_batch += n_batches_per_iter
            val = i_batch, params, loss, opt_state
            return val
        
        i_batch, fps, _, opt_state = jax.lax.while_loop(cond_func, body_func, val)
        
        losses = get_fp_loss_func(func)(fps)
        
        loss_sort_idxs = jnp.argsort(losses)
                
        return i_batch, fps[loss_sort_idxs], losses[loss_sort_idxs]
    
    @eqx.filter_jit
    def find_and_filter(
        self, 
        func: Callable[[Float[Array, "state"]], Float[Array, "state"]], 
        candidates: Float[Array, "n_candidates state"], 
        loss_tol: Float[Array, "1"],   
        n_batches: int = 10_000, 
        n_batches_per_iter: int = 200,
        outlier_tol: Optional[float] = None, 
        unique_tol: Optional[float] = None,
        distance_norm: int | str | None = 2,
        verbose: bool = False,        
    ) -> FPFilteredResults:
        """Finds fixed points and filters by loss, and (optionally) duplicates and outliers."""
        _, fps, losses = self(func, candidates, loss_tol, n_batches, n_batches_per_iter)
        masks, counts = exclude_points(
            fps, losses, loss_tol, outlier_tol, unique_tol, distance_norm, verbose
        )
        return FPFilteredResults(fps, losses, masks, counts)


def exclude_points(
    points: Float[Array, "points dims"],
    values: Float[Array, "points"], 
    value_tol: Float[Array, "1"], 
    outlier_tol: Optional[float] = None, 
    unique_tol: Optional[float] = None,
    ord: int | str | None = 2,
    verbose: bool = False,
):
    """Returns array masks of the elements of `points`:
    
    - whose respective entry in `values` exceeds `value_tol`;
    - whose nearest neighbouring point is more distant than `outlier_tol`;
    - which are duplicates of some already-included point, as determined by
      their distance to that point being less than `unique_tol`.
      
    Arguments:
        points: Array of points to be filtered.
        values: Array of values associated with respective points.
        value_tol: Maximum value for inclusion of a point.
        outlier_tol: Maximum distance to nearest neighbour for inclusion of a 
            point.
        unique_tol: Maximum distance between points to be considered 
            duplicates.    
        ord: The order of the norm used to compute distances. See the 
            documentation for `jax.numpy.linalg.norm`.
        verbose: Whether to report on the number of points excluded by each 
            criterion.
    """
    n = len(points)
    if verbose:
        print(f"Filtering {n} points:")
    
    tol_mask = values < value_tol
    if verbose:
        print("\tPoints with values not meeting tolerance: "
              f"{n - sum(tol_mask)}/{n}")
    
    outliers_mask = duplicates_mask = jnp.ones(n, jnp.bool_)
    
    if (outlier_tol is not None or unique_tol is not None) and len(points) > 1:
        distances = squareform_pdist(points, ord=ord)
        
        if outlier_tol is not None:
            outliers_mask = find_outliers_mask(distances, outlier_tol)
            if verbose: 
                print(f"\tOutliers: {sum(outliers_mask)}/{n}")
        
        if unique_tol is not None:
            duplicates_mask = find_duplicates_mask(distances, unique_tol)
            if verbose: 
                print(f"\tDuplicates: {sum(duplicates_mask)}/{n}")
                
        if verbose:
            print("\n\tNote that these categories are not exclusive, so their "
                  f"sum may exceed {n}.")
                
    elif verbose:
        print(f"\t\nDataset contains {len(points)} points; " 
              "no outlier or duplicate checks were performed.")

    masks = {
        'meets_all_criteria': tol_mask & outliers_mask & duplicates_mask, 
        'meets_tolerance': tol_mask, 
        'not_outlier': outliers_mask, 
        'unique': duplicates_mask,
    }

    counts = jax.tree_map(jnp.sum, masks)
    
    return masks, counts


def squareform_pdist(xs: Float[Array, "points dims"], ord: int | str | None = 2):
    """Return the pairwise distance matrix between points in `x`.
    
    In the case of `ord=2`, this should be equivalent to:
    
        ```python
        from scipy.spatial.distance import pdist, squareform
        
        squareform(pdist(x, metric='euclidean'))
        ```
    
    However, note that the values for `ord` are those supported
    by `jax.numpy.linalg.norm`. This provides fewer metrics than those 
    supported by `scipy.spatial.distance.pdist`.
    """
    dist = lambda x1, x2: jnp.linalg.norm(x1 - x2, ord=ord)
    row_dist = lambda x: jax.vmap(dist, in_axes=(None, 0))(x, xs)
    return jax.lax.map(row_dist, xs)


def find_outliers_mask(arr, tol):
    """Returns mask of columns of `arr` -- false when the min element exceeds `tol`."""
    arr_eyeless = jnp.fill_diagonal(arr, jnp.inf, inplace=False)
    outlier_mask = jnp.any(arr_eyeless <= tol, axis=0)
    return outlier_mask


def find_duplicates_mask(dists, tol, bound: Literal['low', 'high'] = 'low'):
    """Given a distance matrix, return a 1D mask of all duplicate entries,
    keeping `True` one of each set of duplicates.
    
    Whether an entry is a duplicate is determined by whether its distance to its nearest
    neighbour is less than some threshold `tol`. 
    
    Taking the pairwise threshold as a relation, we could partition the points into a set
    of disconnected graphs, where each member of each graph satisfies the duplicate
    relation with at least one other member of that graph. Computing these graphs would
    be relatively complex and annoying, but it would make the result of this function
    invariant to permutations of the distance matrix (i.e. permutations of the points 
    used to compute the distance matrix).  
    
    On the other hand, a pairwise approach to identifying duplicates would tend
    to retain whichever member of the pair appeared first in the distance matrix, such that
    this function would not be permutation invariant. 
    
    We take a middle ground. We start by permuting the distance matrix such that the
    points appear in order of the number of duplicates they have. Sorting them in
    descending order, the points with the most duplicates appear first, and thus we tend
    to exclude the more "peripheral" duplicates. In ascending order, we tend 
    to exclude the more "central" duplicates . At each of these bounds, `'low'` and 
    `'high'` respectively, the result of this function is permutation-invariant. 
    
    !!! Note
        Generally, you should choose `bound='low'`. In cases where all points are 
        duplicates, `bound='high'` may exclude everything. 
    
    !!! Example 
        ```python
        xs = jnp.array([
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
        ])    
        dists = squareform_pdist(xs)

        find_duplicates_mask(dists, 0.2, bound='low')
        # array([False, True, False])

        find_duplicates_mask(dists, 0.2, bound='high')
        # array([True, False, True])
        ```
    
    """
    d = {'low': -1, 'high': 1}[bound]
    
    idx_by_n_dups = jnp.argsort(jnp.sum(dists < tol, axis=0))[::d]
    invert_idx_by_n_dups = jnp.argsort(idx_by_n_dups)
    dists = dists[:, idx_by_n_dups][idx_by_n_dups, :]
    
    tril_idxs = jnp.tril_indices(dists.shape[0], 0, dists.shape[1])
    dists_triu = dists.at[tril_idxs].set(jnp.inf)
    duplicate_mask = jnp.all(dists_triu > tol, axis=0)
    
    return duplicate_mask[invert_idx_by_n_dups]


def fp_adam_optimizer(
    learning_rate: float = 0.2, 
    decay_steps: int = 1, 
    decay_rate: float = 0.9999, 
    b1: float = 0.9, 
    b2: float = 0.999, 
    eps: float = 1e-5,
):
    """Default Adam optimizer for finding fixed points.
    
    Incorporates an exponential learning rate decay schedule.
    """
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
    )
    optimizer = optax.adam(
        learning_rate=schedule, b1=b1, b2=b2, eps=eps,
    )
    return optimizer
    

def get_fp_loss_func(func):
    """Returns the batched fixed point MSE loss for a given function."""
    batch_func = jax.vmap(func, in_axes=(0,))
    def loss_func(x):
        return jnp.mean((x - batch_func(x)) ** 2, axis=1)
    return loss_func


def get_total_fp_loss_func(func):
    loss_func = get_fp_loss_func(func)
    def total_loss_func(x):
        return jnp.mean(loss_func(x))
    return total_loss_func


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
    fps: Float[Array, "condition *fp state"], 
    # Unfortunately there's no way to type the shape of sklearn's PCA
    pca: PCA, 
    colors: str | Sequence[str] | None = None, 
    candidates_alpha: float = 0.05, 
    marker_size: int = 3, 
    marker_symbol: str = 'circle',
    #! Is this why we can't plot all the FPs in nb6?
    n_plot_max: int = 1000, 
    fp_candidates: Optional[Float[Array, "candidate state"]] = None, 
    label: str = '',
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    if fp_candidates is not None:
        candidates_pc = pca.transform(np.array(fp_candidates))
        emax = candidates_pc.shape[0] if candidates_pc.shape[0] < n_plot_max else n_plot_max
    else:
        candidates_pc = None
        emax = n_plot_max

    fps_flat = np.reshape(fps, (-1, fps.shape[-1]))
    n_fps_per_condition = np.prod(fps.shape[1:-1])
    fps_flat_pc = pca.transform(fps_flat)
    
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
    
    if label != '':
        label = f' ({label})'
    
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
                    name=f'Fixed points{label}',
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
    fig = fbp.plot_traj_3D(trajs_pcs, colors=colors, fig=fig)
    
    return fig


N_KEEP_MAX = 10


def take_top_fps(
    fpf_results: PyTree[FPFilteredResults], 
    n_keep: int = 1,
    warn: bool = True, 
):
    """Slice out the leading FPs.
    
    Argument:
        fpf_results: Tree of fixed point finder results, containing all optimized candidate fixed points.
        n_keep: The number of top FPs to keep. The maximum number that can be returned is 10.
        warn: Whether to log a warning if some qualifying FPs are excluded from the result.
    """
    #! I'm not sure what happens if `n_keep` is greater than the number of qualifying FPs!
    
    @partial(jnp.vectorize, signature='(n,m),(n)->(p,m)')
    def take_top_fp(fps, mask):
        # Get the actual count of True values
        n_matches = jnp.sum(mask)
        # Get fixed-size indices using size parameter
        idxs = jnp.where(mask, size=N_KEEP_MAX)[0]  # Or whatever max size needed
        # Create a mask for valid indices (< n_matches and < n_keep)
        idx_mask = jnp.arange(n_keep) < jnp.minimum(n_matches, n_keep)
        # Create output array of fixed size n_keep
        result = jnp.full((n_keep, fps.shape[1]), jnp.nan)
        # Only set values where idx_mask is True
        result = result.at[jnp.arange(n_keep)].set(
            jnp.where(
                idx_mask[..., None],  # Shape will be (5, 1) to broadcast with (5, 100)
                fps[idxs][:n_keep],
                result
            )
        )
        return result
    
    if warn:
        # get the idxs of any conditions that have more or less than 1 FP 
        idxs_multi_fps = jt.map(
            lambda results: np.where(results.counts['meets_all_criteria'] > n_keep), 
            fpf_results, 
            is_leaf=is_module,
        )
        # TODO: Warn how many FPs, and give the indices
        if np.any(np.concatenate(jt.leaves(idxs_multi_fps))):
            logger.warn("Some FPs satisfied criteria but were omitted")

    return jt.map(
        lambda results: take_top_fp(results.fps, results.masks['meets_all_criteria']), 
        fpf_results, 
        is_leaf=is_module,
    )