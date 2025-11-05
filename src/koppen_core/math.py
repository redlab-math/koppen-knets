
import jax.numpy as jnp
from jax import jit



@jit
def beta_jax(n: int, r: int) -> float:
    """JAX version of beta function."""
    return jnp.where(
        n == 1,
        float(r),
        (jnp.power(float(n), float(r)) - 1.0) / (float(n) - 1.0)
    )

def compute_lambdas(n: int) -> jnp.ndarray:
    """Computes rationally independent weights."""
    lambdas = jnp.array([1.0 / jnp.sqrt(p + 2) for p in range(n)])
    return lambdas / jnp.sum(lambdas)


def beta_cpu(n: int, r: int) -> float:
    """Pure Python/NumPy version of beta function for CPU execution."""
    if n == 1:
        return float(r)
    return (n**r - 1.0) / (n - 1.0)



def default_epsilon(n: int) -> float:
    """Actor's default shift parameter: ε = 1/(2n)"""
    return 1.0 / (2.0 * n)

def default_gamma(n: int) -> int:
    """Actor uses γ = 2n + 2 as base."""
    return 2 * n + 2
