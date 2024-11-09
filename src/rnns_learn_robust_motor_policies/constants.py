from feedbax.intervene import (
    CurlField, 
    FixedField, 
)


INTERVENOR_LABEL = "DisturbanceField"
DISTURBANCE_CLASSES = {
    'curl': CurlField,
    'random': FixedField,
}
MASS = 1.0
WORKSPACE = ((-1., -1.),
             (1., 1.))
