"""
Compute derivatives of functions. 
"""

from collections.abc import Sequence
from types import MappingProxyType
from typing import ClassVar, Optional

import jax
import jax.tree as jt

from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisDefaultInputsType, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace, RequiredInput


#! TODO: `Jacobians` and `Hessians` seem like good candidates for 
#! refactoring by a simpler `AbstractAnalysis` functional constructor
class Jacobians(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        funcs=RequiredInput,
        func_args=RequiredInput,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    argnums: Optional[Sequence[int]] = None

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args,
        **kwargs,
    ):
        def get_jacs(func, args):
            if self.argnums is None:
                argnums = tuple(range(len(args)))
            else:
                argnums = self.argnums

            return jax.jacobian(func, argnums=argnums)(*args)

        return jt.map(get_jacs, funcs, func_args)


class Hessians(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        funcs=RequiredInput,
        func_args=RequiredInput,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    argnums: Optional[Sequence[int]] = None
    diag_only: bool = True  # Whether to compute only the diagonal Hessians (i.e. no cross-input terms)

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args,
        **kwargs,
    ):
        def get_hessians(func, args):
            # Conditional is fine because it always happens at compile time
            if self.argnums is None:
                argnums = tuple(range(len(args)))
            else:
                argnums = self.argnums

            if self.diag_only:
                return tuple(jax.hessian(func, argnums=i)(*args) for i in argnums)
            else:
                return jax.hessian(func, argnums=argnums)(*args)

        return jt.map(get_hessians, funcs, func_args)