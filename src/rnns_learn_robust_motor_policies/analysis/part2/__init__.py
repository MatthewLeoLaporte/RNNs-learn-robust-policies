
from .unit_perts import (
    ALL_ANALYSES as ANALYSES_UNIT_PERTS,
    COLOR_FUNCS as COLOR_FUNCS_UNIT_PERTS,
    setup_eval_tasks_and_models as setup_unit_perts,
    eval_func as eval_func_unit_perts,
)

ANALYSIS_SETS = {
    "2-1": None,
    "2-2": None,
    "2-3": None,
    "2-4": None,
    "2-5": None,
    "2-6": (setup_unit_perts, eval_func_unit_perts, ANALYSES_UNIT_PERTS, COLOR_FUNCS_UNIT_PERTS),
}