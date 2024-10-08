

import jax.numpy as jnp 


def angle_between_vectors(v2, v1):
    """Return the signed angle between two 2-vectors."""
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0], 
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )   
    

def forward_lateral_vels(velocity, trial_specs):
    init_pos = trial_specs.inits['mechanics.effector'].pos
    goal_pos = trial_specs.targets['mechanics.effector.pos'].value[:, -1]

    # angles = jnp.arctan2(
    #     goal_pos[..., 1] - init_pos[..., 1], 
    #     goal_pos[..., 0] - init_pos[..., 0]
    # )
    trial_vec = goal_pos - init_pos

    speeds = jnp.sqrt(jnp.sum(velocity ** 2, axis=-1))

    rel_angles = angle_between_vectors(velocity, trial_vec[:, None])

    forward = speeds * jnp.cos(rel_angles)
    lateral = speeds * jnp.sin(rel_angles)
    
    return forward, lateral