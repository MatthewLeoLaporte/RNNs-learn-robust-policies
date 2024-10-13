

import jax.numpy as jnp 
from jaxtyping import Array, Float


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
    line_vec = goal_pos - init_pos
    
    # Normalize the line vector
    line_vec_norm = line_vec / jnp.linalg.norm(line_vec, axis=-1, keepdims=True)
    
    # Broadcast line_vec_norm to match velocity's shape
    line_vec_norm = line_vec_norm[:, None]  # Shape: (conditions, 1, xy)
    
    # Calculate forward velocity (dot product)
    forward = jnp.sum(velocity * line_vec_norm, axis=-1)
    
    # Calculate lateral velocity (cross product)
    lateral = jnp.cross(line_vec_norm, velocity)
    
    return jnp.stack([forward, lateral], axis=-1)


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
    pos = jnp.swapaxes(pos, -3, -2)
    
    # Calculate the vectors from 1) inits to goals, and 2) inits to trajectory positions
    line_vec = goal_pos - init_pos
    point_vec = pos - init_pos

    # Calculate the cross product between the line vector and the point vector
    # This is the area of the parallelogram they form.
    cross_product = jnp.cross(line_vec, point_vec)
    
    # Obtain the parallelogram heights (i.e. the lateral distances) by dividing 
    # by the length of the line vectors.
    line_length = jnp.linalg.norm(line_vec, axis=-1)
    lateral_dist = jnp.abs(cross_product) / line_length

    return jnp.swapaxes(lateral_dist, -2, -1)
