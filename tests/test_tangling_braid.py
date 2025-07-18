"""
Demonstrate the discrete measure of trajectory tangling.

Generates two trajectories:

1. A Lissajous/braid-like trajectory with several crossovers.
2. A slowly decaying circular (limit cycle) trajectory. 

Computes the discrete measure of tangling for all the lumped states. 

Plots the trajectories, coloured by the tangling value. 

Note that tangling increases as we approach a crossover point in the braid,
where it reaches a very high maximum value. On the other hand, all states in 
the circle have approximately the same, near-zero tangling value, as they 
are far from states flowing in a different direction from them. 
"""


import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go

from rlrmp.analysis.tangling import _get_tangling_core


def generate_braid(T: int = 2000, freq_x: int = 2, freq_y: int = 3, phase: float = 0.5):
    """Generate a 2-D Lissajous/braid-like trajectory with many crossovers."""
    t = jnp.linspace(0, 2 * jnp.pi, T)
    x = jnp.sin(freq_x * t)
    y = jnp.sin(freq_y * t + phase) / 2.0  # scale y to make loops visible
    dt = float(t[1] - t[0])
    return jnp.stack([x, y], axis=-1), dt


def generate_decaying_circle(
    T: int,
    center: tuple[float, float] = (0.5, 0.0),
    radius: float = 0.25,
    n_orbits: int = 3,
    decay_per_orbit: float = 0.01,
):
    """Return points on a slightly contracting circle over multiple orbits."""

    t = jnp.linspace(0.0, 2 * jnp.pi * n_orbits, T)

    # Linear decay of radius with angle: after each full orbit radius decays
    # by `decay_per_orbit`.
    r = radius * (1.0 - decay_per_orbit * (t / (2 * jnp.pi)))

    x = center[0] + r * jnp.cos(t)
    y = center[1] + r * jnp.sin(t)
    return jnp.stack([x, y], axis=-1)


def flow_field(coords: jnp.ndarray, dt: float):
    prev = coords[:-1]
    nxt = coords[1:]
    return prev, (nxt - prev) / dt  # align lengths


def main():
    # ---------------- Trajectories ----------------------------------------
    coords_braid, dt = generate_braid()
    coords_circle = generate_decaying_circle(coords_braid.shape[0], radius=0.25)

    # Stack so shape -> (2, T, 2)
    coords_all = jnp.stack([coords_braid, coords_circle])

    # Flow field (finite diff) along time axis (-2)
    prev = coords_all[:, :-1, :]
    nxt = coords_all[:, 1:, :]
    dcoords_dt = (nxt - prev) / dt  # shape (2, T-1, 2)

    tangling_core = _get_tangling_core(
        eps=1e-6,
        method="kdtree",
        k_neigh=30,
        leaf_size=40,
        t_axis=-2,
    )
    tangling = tangling_core(prev, dcoords_dt)  # shape (2, T-1)

    # Convert to NumPy for plotting
    tang_np = np.asarray(tangling)
    coords_np = np.asarray(coords_all)

    # ---------------- Plot -------------------------------------------------
    fig = go.Figure()

    colorscale = "Plasma"
    labels = ["braid", "circle"]

    cmin_global = tang_np.min()
    cmax_global = tang_np.max()

    for idx in range(2):
        fig.add_trace(
            go.Scatter(
                x=coords_np[idx, :-1, 0],
                y=coords_np[idx, :-1, 1],
                mode="markers+lines",
                name=labels[idx],
                marker=dict(
                    size=4,
                    color=tang_np[idx],
                    colorscale=colorscale,
                    cmin=cmin_global,
                    cmax=cmax_global,
                    colorbar=dict(title="Tangling") if idx == 0 else None,
                    showscale=True if idx == 0 else False,
                ),
                line=dict(color="lightgrey"),
                hovertemplate="x:%{x:.3f}<br>y:%{y:.3f}<br>Tangling:%{marker.color:.2e}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Braid + circle trajectories coloured by discrete tangling",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
    )
    fig.show()
    return fig


if __name__ == "__main__":
    main() 