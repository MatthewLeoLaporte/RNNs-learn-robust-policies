

import equinox as eqx
import jax.numpy as jnp 
import jax.random as jr
from jaxtyping import Array, Float

from feedbax import is_type
from feedbax.intervene import AbstractIntervenor
from feedbax._tree import tree_infer_batch_size


def angle_between_vectors(v2, v1):
    """Return the signed angle between two 2-vectors."""
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0], 
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )   


def get_forward_lateral_vel(
    velocity: Float[Array, "*batch conditions time xy=2"], 
    pos_endpoints: Float[Array, "point=2 conditions xy=2"],
) -> Float[Array, "*batch conditions time 2"]:
    """Given x-y velocity components, rebase onto components forward and lateral to the line between endpoints.
    
    Arguments:
        velocity: Trajectories of velocity vectors.
        pos_endpoints: Initial and goal reference positions for each condition, defining reference lines.
    
    Returns:
        forward: Forward velocity components (parallel to the reference lines).
        lateral: Lateral velocity components (perpendicular to the reference lines).
    """
    init_pos, goal_pos = pos_endpoints
    direction_vec = goal_pos - init_pos
    
    return project_onto_direction(velocity, direction_vec)
    

def project_onto_direction(
    var: Float[Array, "*batch conditions time xy=2"],
    direction_vec: Float[Array, "conditions xy=2"],
):
    """Projects components of arbitrary variables into components parallel and orthogonal to a given direction.
    
    Arguments:
        var: Data with x-y components to be projected. 
        direction_vector: Direction vectors. 
    
    Returns:
        projected: Projected components (parallel and orthogonal).
    """
    # Normalize the line vector
    direction_vec_norm = direction_vec / jnp.linalg.norm(direction_vec, axis=-1, keepdims=True)
    
    # Broadcast line_vec_norm to match velocity's shape
    direction_vec_norm = direction_vec_norm[:, None]  # Shape: (conditions, 1, xy)
    
    # Calculate forward velocity (dot product)
    parallel = jnp.sum(var * direction_vec_norm, axis=-1)
    
    # Calculate lateral velocity (cross product)
    orthogonal = jnp.cross(direction_vec_norm, var)
    
    return jnp.stack([parallel, orthogonal], axis=-1)


def get_lateral_distance(
    pos: Float[Array, "*batch conditions time xy=2"], 
    pos_endpoints: Float[Array, "point=2 conditions xy=2"],
) -> Float[Array, "*batch conditions time"]:
    """Compute the lateral distance of points from the straight line connecting init and goal positions.
    
    Arguments:
        pos: Trajectories of positions.
        pos_endpoints: Initial and goal reference positions for each condition.
    
    Returns:
        Trajectories of lateral distances to the straight line between endpoints.
    """
    init_pos, goal_pos = pos_endpoints
    
    # Rearrange input axes to ensure proper broadcasting over conditions
    # pos = jnp.swapaxes(pos, -3, -2)
    
    # Calculate the vectors from 1) inits to goals, and 2) inits to trajectory positions
    direction_vec = goal_pos - init_pos
    point_vec = pos - init_pos[..., None, :]

    # Calculate the cross product between the line vector and the point vector
    # This is the area of the parallelogram they form.
    cross_product = jnp.cross(direction_vec[..., None, :], point_vec)
    
    # Obtain the parallelogram heights (i.e. the lateral distances) by dividing 
    # by the length of the line vectors.
    line_length = jnp.linalg.norm(direction_vec, axis=-1)
    # lateral_dist = jnp.abs(cross_product) / line_length
    lateral_dist = jnp.abs(cross_product) / line_length[..., None]

    return lateral_dist
    # return jnp.swapaxes(lateral_dist, -2, -1)


def _get_eval_ensemble(models, task):
    def eval_ensemble(key):
        return task.eval_ensemble(
            models,
            n_replicates=tree_infer_batch_size(models, exclude=is_type(AbstractIntervenor)),
            ensemble_random_trials=False,
            key=key,
        )
    return eval_ensemble

    
@eqx.filter_jit
def vmap_eval_ensemble(models, task, n_trials: int, key):
    """Evaluate an ensemble of models on `n` random trials of a task."""
    return eqx.filter_vmap(_get_eval_ensemble(models, task))(
        jr.split(key, n_trials)
    )