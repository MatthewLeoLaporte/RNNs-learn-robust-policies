import jax.tree as jt

from feedbax.intervene import CurlField, FixedField, add_intervenors, schedule_intervenor
from jax_cookbook import is_module 
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.state_utils import orthogonal_field
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.setup_utils import convert_tasks_to_small
from rnns_learn_robust_motor_policies.types import PertAmpDict


DISTURBANCE_FUNCS = {
    'curl': lambda amplitude: CurlField.with_params(amplitude=amplitude),
}


def setup_tasks_and_models(models, task_base, hps):
    if hps.disturbance.type == 'curl':   
        def disturbance(amplitude):
            return CurlField.with_params(amplitude=amplitude)    
            
    elif hps.disturbance.type == 'constant':    
        def disturbance(amplitude):           
            return FixedField.with_params(
                scale=amplitude,
                field=orthogonal_field,  
            ) 
            
    else:
        raise ValueError(f"Unknown disturbance type: {hps.disturbance.type}")

    # Insert the disturbance field component into each model
    models = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models,
        is_leaf=is_module,
    )

    all_tasks = dict()

    # Assume a sequence of amplitudes is provided, as in the default config
    disturbance_amplitudes = hps.disturbance.amplitude
    # Generate tasks with different amplitudes of disturbance field
    all_tasks['full'], _ = jtree.unzip(jt.map(
        lambda disturbance_amplitude: schedule_intervenor(
            task_base, models[0],
            lambda model: model.step.mechanics,
            disturbance(disturbance_amplitude),
            label=INTERVENOR_LABEL,  
            default_active=False,
        ),
        PertAmpDict(zip(disturbance_amplitudes, disturbance_amplitudes)),
    ))

    # Make smaller versions of the tasks for visualization.
    all_tasks['small'] = convert_tasks_to_small(all_tasks['full'])
    
    return all_tasks, models