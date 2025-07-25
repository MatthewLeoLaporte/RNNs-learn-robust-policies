# tests/test_analysis_vmap.py

import numpy as np
import jax.numpy as jnp
from types import MappingProxyType
from dataclasses import fields
from typing import ClassVar, Optional

import pytest

from rlrmp.analysis.analysis import (
    AbstractAnalysis,
    AnalysisInputData,
    AnalysisDefaultInputsType,
    DefaultFigParamNamespace,
    FigParamNamespace,
)
from jax_cookbook import MultiVmapAxes


class AddAnalysis(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType({})
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    def compute(self, data: AnalysisInputData, *, x, y, **kwargs):
        return x + y


# A “blank” AnalysisInputData to satisfy the first positional arg:
_dummy_data = AnalysisInputData(**{
    fld.name: MappingProxyType({}) for fld in fields(AnalysisInputData)
})


def test_plain_compute():
    a = AddAnalysis()
    out = a._compute_with_ops(_dummy_data, x=2, y=3)
    assert out == 5


def test_elementwise_vectorize_axis0():
    # Both x and y are arrays of same length → elementwise sum
    a = AddAnalysis().vmap({"x": 0, "y": 0})
    x = jnp.array([1, 2, 3])
    y = jnp.array([10, 20, 30])
    out = a._compute_with_ops(_dummy_data, x=x, y=y)
    np.testing.assert_array_equal(out, x + y)


def test_nested_vmap_via_MultiVmapAxes_ordering():
    # MultiVmapAxes(0,1) means inner vmap over axis 0, then outer over axis 1
    a = AddAnalysis().vmap({
        "x": MultiVmapAxes(0, 1),
        "y": MultiVmapAxes(0, 1),
    })
    x = jnp.array([[1, 2, 3],
                   [4, 5, 6]])   # shape (2,3)
    y = jnp.array([[10,20,30],
                   [40,50,60]])  # shape (2,3)

    out = a._compute_with_ops(_dummy_data, x=x, y=y)

    # inner vmap (axis=0) gives shape (2,3); then outer (axis=1) gives (3,2)
    assert out.shape == (3, 2)
    # values should be (x+y).T
    np.testing.assert_array_equal(out, (x + y).T)


def test_successive_vmap_equiv_nested():
    # a1 maps axis0 then axis1; a2 uses MultiVmapAxes(0,1) — should match
    a1 = AddAnalysis().vmap({"x": 0, "y": 0}).vmap({"x": 1, "y": 1})
    a2 = AddAnalysis().vmap({
        "x": MultiVmapAxes(0, 1),
        "y": MultiVmapAxes(0, 1),
    })

    x = jnp.array([[2, 4], [6, 8]])  # (2,2)
    y = jnp.array([[1, 3], [5, 7]])  # (2,2)

    out1 = a1._compute_with_ops(_dummy_data, x=x, y=y)
    out2 = a2._compute_with_ops(_dummy_data, x=x, y=y)
    np.testing.assert_array_equal(out1, out2)


def test_swapped_vmap_order_errors():
    # Mapping axis1 first then axis0 should error on the second vmap
    a = AddAnalysis().vmap({"x": 1, "y": 1})
    with pytest.raises(Exception):
        _ = a.vmap({"x": 0, "y": 0})._compute_with_ops(_dummy_data,
                                                       x=jnp.array([[1,2],[3,4]]),
                                                       y=jnp.array([[5,6],[7,8]]))