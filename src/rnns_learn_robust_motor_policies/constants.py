from feedbax.intervene import (
    CurlField, 
    FixedField, 
)


INTERVENOR_LABEL = "DisturbanceField"
DISTURBANCE_CLASSES = {
    'curl': CurlField,
    'random': FixedField,
}