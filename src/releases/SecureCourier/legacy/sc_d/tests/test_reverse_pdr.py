from __future__ import annotations

import numpy as np

from reverse_pdr import reverse_solve_arrays


def micro_inputs():
    return {
        "pair_state": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "pair_step": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "successor": np.asarray([1, 1, -1, -1], dtype=np.int32),
        "continuation": np.asarray([True, True, False, False]),
        "state_step": np.asarray([0, 1], dtype=np.int16),
        "pi_ref": np.asarray([0.5, 0.5, 0.5, 0.5]),
        "pi_current": np.asarray([0.5, 0.5, 0.75, 0.25]),
        "qbar": np.asarray([0.0, 0.0, 2.0, 0.0]),
        "max_steps": 2,
    }


def test_hand_computed_micro_dag():
    result = reverse_solve_arrays(**micro_inputs())
    np.testing.assert_allclose(result.source_by_state, [0.0, 0.5], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.correction_by_pair, [0.485, 0.485, 0.0, 0.0], atol=1e-15, rtol=0.0)
    assert result.residual_inf <= 1e-15


def test_zero_policy_change_zero_correction():
    inputs = micro_inputs()
    inputs["pi_current"] = inputs["pi_ref"].copy()
    result = reverse_solve_arrays(**inputs)
    assert np.max(np.abs(result.source_by_state)) == 0.0
    assert np.max(np.abs(result.correction_by_pair)) == 0.0
    assert result.residual_inf == 0.0


def test_terminal_correction_is_zero():
    result = reverse_solve_arrays(**micro_inputs())
    assert np.array_equal(result.correction_by_pair[2:], np.zeros(2))

