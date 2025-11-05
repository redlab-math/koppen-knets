"""
Immutable configuration using NamedTuple (JAX pytree compatible).
"""

from typing import NamedTuple
import jax.numpy as jnp


class KSTConfig(NamedTuple):
    """
    Immutable KST configuration - compatible with JAX pytrees.
    
    All fields are primitives or JAX arrays, NO Python objects with logic.
    """
    n: int              # Spatial dimension
    gamma: int          # Base for digit expansion
    k: int              # Refinement level
    epsilon: float      # Shift parameter ε ∈ (0, 1/(2n)]
    
    @property
    def num_points(self) -> int:
        """Total points to tabulate: γ^k + 1"""
        return self.gamma ** self.k + 1
    
    @property
    def Q(self) -> int:
        """Number of outer functions: 2n + 1"""
        return 2 * self.n + 1
    
    def __repr__(self) -> str:
        return (f"KSTConfig(n={self.n}, γ={self.gamma}, k={self.k}, "
                f"ε={self.epsilon:.6f}, points={self.num_points:,})")


def create_default_config(n: int, k: int, 
                         gamma: int = None, 
                         epsilon: float = None) -> KSTConfig:
    """
    Create configuration with Actor's defaults.
    
    Defaults:
        gamma = 2n + 2 (Appendix C)
        epsilon = 1/(2n) (Section 3.2)
    """
    if gamma is None:
        gamma = 2 * n + 2
    
    if epsilon is None:
        epsilon = 1.0 / (2.0 * n)
    
    return KSTConfig(n=n, gamma=gamma, k=k, epsilon=epsilon)


def validate_config(config: KSTConfig) -> None:
    """
    Validate configuration against Actor's constraints.
    
    Constraints:
    1. n ≥ 1 (dimension)
    2. k ≥ 1 (refinement)
    3. γ > 2n + 1 (base)
    4. ε ∈ (0, 1/(2n)] (shift)
    """
    errors = []
    
    if config.n < 1:
        errors.append(f"Dimension n={config.n} must be ≥ 1")
    
    if config.k < 1:
        errors.append(f"Refinement k={config.k} must be ≥ 1")
    
    if config.gamma <= 2 * config.n + 1:
        errors.append(f"Base γ={config.gamma} must be > 2n+1 = {2*config.n+1}")
    
    max_epsilon = 1.0 / (2.0 * config.n)
    if not (0 < config.epsilon <= max_epsilon):
        errors.append(f"Shift ε={config.epsilon} must be in (0, {max_epsilon:.6f}]")
    
    if errors:
        raise ValueError("Invalid KST configuration:\n  " + "\n  ".join(errors))