from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import equinox as eqx
from equinox import AbstractVar, Module
import jax.tree as jt
from jax_cookbook import is_module
from jaxtyping import PyTree, Array
import plotly.graph_objects as go

import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.measures import Responses
from rnns_learn_robust_motor_policies.analysis.state_utils import get_aligned_vars, get_pos_endpoints
from rnns_learn_robust_motor_policies.database import add_evaluation_figure
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.misc import camel_to_snake, get_dataclass_fields
from rnns_learn_robust_motor_policies.plot_utils import figs_flatten_with_paths
from rnns_learn_robust_motor_policies.tree_utils import tree_level_types
from rnns_learn_robust_motor_policies.types import TYPE_LABELS  

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


PLANT_VAR_LABELS = ('position', 'velocity', 'force')
WHERE_PLOT_PLANT_VARS = lambda states: Responses(
    states.mechanics.effector.pos,
    states.mechanics.effector.vel,
    states.efferent.output,
)


WHERE_VARS_TO_ALIGN = lambda states, pos_endpoints: Responses(
    # Positions with respect to the origin
    states.mechanics.effector.pos - pos_endpoints[0][..., None, :],
    states.mechanics.effector.vel,
    states.efferent.output,
)

class AbstractAnalysis(Module):
    """Component in an analysis pipeline.
    
    In `run_analysis`, multiple sets of evaluations may be performed
    prior to analysis. In particular, we may evaluate a full/large set
    of task conditions for statistical purposes, and evaluate a smaller 
    version for certain visualizations. Thus `AbstractAnalysis` 
    subclasses expect arguments `models`, `tasks`, `states`, and `hps` all 
    of which are PyTrees (in practice I have only done a single dict level
    to contain the eval variants, but typed it more generall as a PyTree).
    
    - Inside `make_figs` and `compute` we need to be explicit about the variant;
      e.g. refer to `states['full']` when analyzing the larger eval set.
    - The `variant` field indicates which 
    
    Now, while it may be the case that an analysis would depend on both the 
    larger and smaller variants (in our example), we still must specify only a 
    single `variant`, since this determines the hyperparameters that are passed 
    to `analysis.save`. Thus it is assumed that all figures that result from a 
    call to some `AbstractAnalysis.make_figs` will be best associated with only
    one (and always the same one) of the eval variants.
    
    TODO: If we return the hps on a fig-by-fig basis from within `make_figs`, then 
    we could avoid this limitation.    
    
    Abstract class attributes:
        dependencies: Specifies the subclasses of `AbstractAnalysis`
            whose results are needed for this subclass of `AbstractAnalysis`.
        variant: Label of the evaluation variant this analysis uses (primarily).
    
    Abstract fields:
        conditions: In `run_analysis`, certain condition checks are performed. The 
            analysis is only run if all of the checks whose keys are in `conditions`
            are successful. For example, certain figures may only make sense to generate
            when there is system noise (i.e. multiple evals per condition), and in 
            that case we could give the condition `"any_system_noise"` to those analyses.
    """
    dependencies: AbstractClassVar[MappingProxyType[str, "type[AbstractAnalysis]"]]
    variant: AbstractClassVar[Optional[str]]  #! TODO: Should be an instance var so user can change it
    conditions: AbstractVar[tuple[str, ...]]
    
    def __call__(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace],
        **kwargs,
    ) -> tuple[PyTree[Array], PyTree[go.Figure]]:
        result = self.compute(models, tasks, states, hps, **kwargs)
        figs = self.make_figs(models, tasks, states, hps, result=result, **kwargs)
        return result, figs
        
    def compute(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace],
        **kwargs,
    ) -> Optional[PyTree[Array]]:
        return 
    
    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace],
        *,
        result: Optional[Any],
        **kwargs,
    ) -> Optional[PyTree[go.Figure]]:
        return 
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        """Additional parameters to save.
        
        Note that `**kwargs` here may not only contain the dependencies, but that `save` 
        passes the key-value pairs of parameters inferred from the `figs` PyTree. 
        Thus for example `disturbance_std` is explicitly referred to in the argument list of 
        `plant_perts.CenterOutByEval._params_to_save`.
        """
        return dict()

    def save(self, db_session, eval_info, result, figs, hps, model_info=None, **dependencies):
        param_keys = tuple(TYPE_LABELS[t] for t in tree_level_types(figs))
        
        for path, fig in figs_flatten_with_paths(figs):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))
            
            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **self._field_params,  # From the fields of this subclass
                **self._params_to_save(
                    hps, 
                    result=result, 
                    **path_params, 
                    **dependencies, # Extras specified by the subclass
                ),  
                eval_n=hps.eval_n,  # Some things should always be included?
            )
            
            add_evaluation_figure(
                db_session, 
                eval_info, 
                fig, 
                camel_to_snake(self.__class__.__name__), 
                model_records=model_info, 
                **params,
            )
    
    @cached_property
    def _field_params(self):
        return get_dataclass_fields(self, exclude=('dependencies', 'conditions'))


class AlignedVars(AbstractAnalysis):
    """Align spatial variable (e.g. position and velocity) coordinates with the reach direction."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: ClassVar[Optional[str]] = None
    conditions: tuple[str, ...] = ()

    def compute(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        trial_specs,
        **kwargs,
    ):
        pos_endpoints = jt.map(get_pos_endpoints, trial_specs, is_leaf=is_module)

        return {
            variant: jt.map(
                lambda all_states: get_aligned_vars(all_states, WHERE_VARS_TO_ALIGN, pos_endpoints[variant]),
                states[variant],
                is_leaf=is_module,
            )
            for variant in states
        }



            

