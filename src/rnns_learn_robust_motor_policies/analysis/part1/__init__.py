from .plant_perts import (
    ALL_ANALYSES as ANALYSES_PLANT_PERTS,
    COLOR_FUNCS as COLOR_FUNCS_PLANT_PERTS,
    setup_eval_tasks_and_models as setup_plant_perts,
    eval_func as eval_func_plant_perts,
)

from .feedback_perts import (
    ALL_ANALYSES as ANALYSES_FEEDBACK_PERTS,
    COLOR_FUNCS as COLOR_FUNCS_FEEDBACK_PERTS,
    setup_eval_tasks_and_models as setup_feedback_perts,
    eval_func as eval_func_feedback_perts,
)

from .freq_response import (
    ALL_ANALYSES as ANALYSES_FREQ_RESPONSE,
    COLOR_FUNCS as COLOR_FUNCS_FREQ_RESPONSE,
    setup_eval_tasks_and_models as setup_freq_response,
    eval_func as eval_func_freq_response,
)


ANALYSIS_SETS = {
    "1-1": (setup_plant_perts, eval_func_plant_perts, ANALYSES_PLANT_PERTS, COLOR_FUNCS_PLANT_PERTS), 
    "1-2": (setup_feedback_perts, eval_func_feedback_perts, ANALYSES_FEEDBACK_PERTS, COLOR_FUNCS_FEEDBACK_PERTS),
    "1-3": (setup_freq_response, eval_func_freq_response, ANALYSES_FREQ_RESPONSE, COLOR_FUNCS_FREQ_RESPONSE),
}