

from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble


COLOR_FUNCS = dict()

ALL_ANALYSES = []


def setup_eval_tasks_and_models(task_base, models_base, hps):
    return task_base, models_base, hps


eval_func = vmap_eval_ensemble