
from .unit_perts import (
    ALL_ANALYSES as ANALYSES_UNIT_PERTS,
    COLOR_FUNCS as COLOR_FUNCS_UNIT_PERTS,
    setup_eval_tasks_and_models as setup_unit_perts,
    eval_func as eval_func_unit_perts,
)

from .context_pert import (
    ALL_ANALYSES as ANALYSES_CONTEXT_PERT,
    COLOR_FUNCS as COLOR_FUNCS_CONTEXT_PERT,
    setup_eval_tasks_and_models as setup_context_pert,
    eval_func as eval_func_context_pert,
)

ANALYSIS_SETS = {
    "2-1": None,
    "2-2": None,
    "2-3": None,
    "2-4": None,
    "2-5": None,
    "2-6": (setup_unit_perts, eval_func_unit_perts, ANALYSES_UNIT_PERTS, COLOR_FUNCS_UNIT_PERTS),
    "2-7": (setup_context_pert, eval_func_context_pert, ANALYSES_CONTEXT_PERT, COLOR_FUNCS_CONTEXT_PERT),
}