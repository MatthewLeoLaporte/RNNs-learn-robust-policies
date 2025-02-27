from rnns_learn_robust_motor_policies.analysis.state_utils import orthogonal_field


from feedbax.intervene import CurlField, FixedField


DISTURBANCE_FUNCS = {
    'curl': lambda amplitude: CurlField.with_params(
        amplitude=amplitude,
    ),
    'constant': lambda amplitude: FixedField.with_params(
        scale=amplitude,
        field=orthogonal_field,
    ),
}