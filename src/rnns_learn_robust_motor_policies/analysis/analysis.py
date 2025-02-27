from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, Dict, Type, Tuple
from pathlib import Path
import yaml

import equinox as eqx
from equinox import AbstractVar, Module
import jax.tree as jt
from jaxtyping import PyTree, Array
import plotly.graph_objects as go

from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.database import add_evaluation_figure, savefig
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.misc import camel_to_snake, get_dataclass_fields
from rnns_learn_robust_motor_policies.plot_utils import figs_flatten_with_paths
from rnns_learn_robust_motor_policies.tree_utils import tree_level_types
from rnns_learn_robust_motor_policies.types import TYPE_LABELS

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


# Define a representer for objects PyYAML doesn't know how to handle
def represent_undefined(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
yaml.add_representer(object, represent_undefined)


class AbstractAnalysis(Module):
    """Component in an analysis pipeline.
    
    In `run_analysis`, multiple sets of evaluations may be performed
    prior to analysis. In particular, we may evaluate a full/large set
    of task conditions for statistical purposes, and evaluate a smaller 
    version for certain visualizations. Thus `AbstractAnalysis` 
    subclasses expect arguments `models`, `tasks`, `states`, and `hps` all 
    of which are PyTrees. The top-level structure of these PyTrees is always 
    a 
    
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
    variant: AbstractVar[Optional[str]] 
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

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Return kwargs to be used when instantiating dependencies.
        
        Subclasses can override this method to provide parameters for their dependencies.
        Returns a dictionary mapping dependency name to a dictionary of kwargs.
        """
        return {}

    def save_figs(
        self, 
        db_session, 
        eval_info, 
        result, 
        figs: PyTree,   
        hps, 
        model_info=None,
        dump_path: Optional[Path] = None,
        **dependencies,
    ):
        """Save to disk and record in the database each figure in a PyTree of figures, for this analysis.
        """
        param_keys = tuple(TYPE_LABELS[t] for t in tree_level_types(figs))
        
        if dump_path is not None:
            dump_path = Path(dump_path)
            dump_path.mkdir(exist_ok=True, parents=True)
        
        figs_with_paths_flat = figs_flatten_with_paths(figs)
        hps_0 = jt.leaves(hps, is_leaf=is_type(TreeNamespace))[0]
        
        for i, (path, fig) in enumerate(figs_with_paths_flat):
            path_params = dict(zip(param_keys, tuple(jtree.node_key_to_value(p) for p in path)))
            
            params = dict(
                **path_params,  # Inferred from the structure of the figs PyTree
                **self._field_params,  # From the fields of this subclass
                **self._params_to_save(
                    hps_0, 
                    result=result, 
                    **path_params, 
                    **dependencies, # Extras specified by the subclass
                ),  
                eval_n=hps_0.eval_n,  #? Some things should always be included
            )
            
            add_evaluation_figure(
                db_session, 
                eval_info, 
                fig, 
                camel_to_snake(self.__class__.__name__), 
                model_records=model_info, 
                **params,
            )
            
            # Additionally dump to specified path if provided
            if dump_path is not None:                                
                # Create a unique filename
                analysis_name = camel_to_snake(self.__class__.__name__)
                filename = f"{analysis_name}__{i}"
                
                # Save the figure
                savefig(fig, filename, dump_path, ["json"])
                
                # Save parameters as YAML
                params_path = dump_path / f"{filename}.yaml"
                with open(params_path, 'w') as f:
                    yaml.dump(params, f, default_flow_style=False, sort_keys=False)
                    
    @cached_property
    def _field_params(self):
        return get_dataclass_fields(self, exclude=('dependencies', 'conditions'))







            

