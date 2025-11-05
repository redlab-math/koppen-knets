"""
Unit tests for recursion correctness.
"""

import pytest
import jax.numpy as jnp
from koppen_core.recursion import koppen_phi_single, koppen_phi_batch
from koppen_core.math import beta


def test_beta_function():
    """Test β(n, r) computation"""
    assert beta(2, 1) == 2.0
    assert beta(2, 2) == 6.0
    assert beta(1, 5) == 5.0


def test_koppen_boundaries():
    """Test ψ̃(0) = 0 and ψ̃(1) ≈ 1"""
    n, gamma, k = 2, 6, 3
    
    y_0 = koppen_phi_single(0.0, n, gamma, k)
    y_1 = koppen_phi_single(1.0, n, gamma, k)
    
    assert abs(y_0) < 0.01
    assert abs(y_1 - 1.0) < 0.1


def test_koppen_monotonicity():
    """Test strict monotonicity"""
    n, gamma, k = 2, 6, 4
    
    x_test = jnp.linspace(0, 1, 1000)
    y_test = koppen_phi_batch(x_test, n, gamma, k)
    
    diffs = jnp.diff(y_test)
    violations = jnp.sum(diffs < 0)
    
    assert violations == 0, f"Found {violations} monotonicity violations"


def test_koppen_self_similarity():
    """Test that ψ̃_k ≈ ψ̃_{k-1} at coarser resolution"""
    n, gamma = 2, 6
    
    x_test = jnp.linspace(0, 1, 100)
    
    y_k3 = koppen_phi_batch(x_test, n, gamma, 3)
    y_k4 = koppen_phi_batch(x_test, n, gamma, 4)
    
    max_diff = float(jnp.max(jnp.abs(y_k4 - y_k3)))
    
    assert max_diff < 0.5, f"Self-similarity broken: max diff = {max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])