from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

import equinox as eqx
from equinox import Module
from jaxtyping import PyTree, Array
import plotly.graph_objects as go
from tqdm.auto import tqdm

import jax_cookbook.tree as jtree

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


class AbstractAnalysis(Module):
    # TODO: Maybe use the immutable `MappingProxyType` instead of `dict`
    dependencies: AbstractClassVar[dict[str, Callable]]
    conditions: AbstractClassVar[tuple[str, ...]]
    
    def __call__(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: TreeNamespace,
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
        hps: TreeNamespace,
        **kwargs,
    ) -> Optional[PyTree[Array]]:
        return 
    
    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: TreeNamespace,
        *,
        result: Optional[Any],
        **kwargs,
    ) -> Optional[PyTree[go.Figure]]:
        return 
    
    def _params_to_save(self, hps: TreeNamespace, **kwargs):
        """Additional parameters to save.
        
        Note that `**kwargs` here may not only contain the dependencies, but that `save` 
        passes the key-value pairs of parameters inferred from the `figs` PyTree. 
        Thus for example `disturbance_std` is explicitly referred to in the argument list of 
        `plant_perts.CenterOutByEval._params_to_save`.
        """
        return dict()

    def save(self, db_session, eval_info, result, figs, hps, model_info=None):
        param_keys = tuple(TYPE_LABELS[t] for t in tree_level_types(figs))
        
        for path, fig in tqdm(figs_flatten_with_paths(figs)):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))
            
            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **self._field_params,  # From the fields of this subclass
                **self._params_to_save(hps, result=result, **path_params),  # Extras specified by the subclass
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
            

